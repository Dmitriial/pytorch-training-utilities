import json
import logging
import random
import selectors
import sys
from functools import cache
from pathlib import Path
from typing import Protocol, Dict, Optional

import humanize
import numpy as np
import torch
from torch.distributed import broadcast_object_list
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .config import Config
from .distributed import (
    global_leader_only,
    global_rank,
    is_global_leader,
    is_local_leader,
    local_leader_only,
)
from .engines import Engine, Engines, TrainFeeder
from .utils import to_device

_logger = logging.getLogger(__name__)
_engines: Engines
_command: str

# output dynamic data
_writer: Optional[SummaryWriter] = None
_parent = Path(__file__).parent


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path: Optional[str] = 'checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

        self.backup_path = Path(_parent, "../../../../.trainer.json")
        self.first_step = True

    def __call__(self, val_loss, model: Optional = None, save_fn: Optional = None):
        score = abs(val_loss)  # we are moving to 0

        data_state = {}
        if self.first_step:
            if self.backup_path.exists() and self.backup_path.is_file():
                data_state = json.load(self.backup_path.open("r"))
                self.best_score = data_state["best_score"]
                self.counter = data_state["counter"]

            self.first_step = False

        if self.best_score is None:
            self.best_score = score
            if model:
                self.save_checkpoint(val_loss, model)
            elif save_fn is not None:
                save_fn()

        elif score > self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if model:
                self.save_checkpoint(val_loss, model)
            elif save_fn is not None:
                save_fn()

            self.counter = 0

        data_state["best_score"] = self.best_score
        data_state["counter"] = self.counter

        json.dump(data_state, self.backup_path.open("w"))

        return self.early_stop

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


_stopper = EarlyStopping(patience=10)


def get_global_step():
    try:
        return _engines.global_step
    except:
        return None


def get_cfg():
    try:
        return _engines.cfg
    except:
        raise RuntimeError("Trainer has not been setup. Have you called trainer.train?")


def get_cmd():
    try:
        return _command
    except:
        raise RuntimeError("Trainer has not been setup. Have you called trainer.train?")


get_iteration = get_global_step


class EnginesLoader(Protocol):
    def __call__(self) -> Engines:
        ...


def load_engines(engines: dict[str, Engine], config: Config):
    engines = Engines(engines)
    engines.setup(config)
    engines.load_checkpoint()
    return engines


class EvalFn(Protocol):
    def __call__(self, *, engines: Engines):
        ...


class Logger(Protocol):
    def __call__(self, *, data: dict):
        ...


@cache
def _get_stdin_selector():
    selector = selectors.DefaultSelector()
    selector.register(fileobj=sys.stdin, events=selectors.EVENT_READ)
    return selector


def _non_blocking_input():
    global _command
    l = [""]
    if is_global_leader():
        s = ""
        selector = _get_stdin_selector()
        events = selector.select(timeout=0)
        for key, _ in events:
            s: str = key.fileobj.readline().strip()
            _logger.info(f'Get stdin "{s}".')
        l[0] = s
    broadcast_object_list(l, src=0)
    _command = l[0]
    return _command


def _make_infinite_epochs(dl):
    while True:
        _logger.info("New epoch starts.")
        yield from dl


@local_leader_only(default=None)
def logger(data):
    return _logger.info(json.dumps(data, indent=2, default=str))


def seed(seed):
    # Set up random seeds, after fork()
    random.seed(seed + global_rank())
    np.random.seed(seed + global_rank())
    torch.manual_seed(seed + global_rank())


def train(
    engines_loader: EnginesLoader,
    train_dl: DataLoader,
    train_feeder: TrainFeeder,
    eval_fn: EvalFn,
    logger: Logger = logger,
):
    global _writer

    engines = engines_loader()
    cfg = engines.cfg

    if _writer is None:
        _logger.info(f"create writer: {str(cfg.tensorboard_root)}")
        _writer = SummaryWriter(log_dir=str(cfg.tensorboard_root))

    if is_local_leader():
        cfg.dump()
        _logger.info(cfg)

    # Setup global engines
    global _engines
    _engines = engines

    events = []

    eval_fn = global_leader_only(eval_fn)

    # Pre-loop command
    command = _non_blocking_input()
    if command in ["eval", "eval_quit"]:
        engines.eval()
        eval_fn(engines=engines)
        engines.train()
    if command in ["quit", "eval_quit"]:
        return

    # Training loop
    for batch in _make_infinite_epochs(train_dl):
        if engines.global_step >= cfg.max_iter:
            break

        batch = to_device(batch, torch.cuda.current_device())
        stats: Dict = engines.step(feeder=train_feeder, batch=batch)
        elapsed_time = stats.get("elapsed_time", 0)
        logger(data=stats)

        # it's the easiest way to show progress of the model
        if _writer is not None:
            for k, v in stats.items():
                if isinstance(v, str):
                    continue

                _writer.add_scalar(f"{get_cfg().model}/train/{k}", v, global_step=engines.global_step)

        command = _non_blocking_input()

        if "@" in command:
            what, when = command.split("@")
            try:
                events.append((what, int(when)))
                _logger.info(f"Event {command} registered.")
            except Exception as e:
                _logger.error(e)
            command = ""

        # Commands are the current command plus the triggered (i.e. iteration >= trigger point) events
        events = [e for e in events if e[1] >= engines.global_step]
        commands = [command] + [e[0] for e in events if e[1] == engines.global_step]

        for command in commands:
            if command in ["event show", "event"]:
                msg = "Events:\n" + "\n".join(["@".join(map(str, e)) for e in events])
                _logger.info(msg)

            if command == "event clear":
                events.clear()

            if "time" in command:
                target_iter = cfg.max_iter
                if " to " in command:
                    try:
                        target_iter = int(command.split(" to ")[-1])
                    except Exception as e:
                        _logger.error(e)
                remaining_iters = target_iter - engines.global_step + 1
                remaining_time = int(remaining_iters * elapsed_time)
                _logger.info(humanize.precisedelta(remaining_time))

            save_ckpt_every = cfg.save_ckpt_every or cfg.eval_every

            saving_commands = ["save"]

            if cfg.save_on_quit:
                saving_commands.append("quit")

            if engines.global_step != 0 and engines.global_step % save_ckpt_every == 0 or command in saving_commands:
                engines.save_checkpoint(tag=None)

            early_stop = False
            if engines.global_step % cfg.eval_every == 0 or command in ["eval"]:
                engines.eval()
                stats = eval_fn(engines=engines)
                # it's the easiest way to show progress of the model
                if _writer is not None:
                    for k, v in stats.items():
                        if isinstance(v, str):
                            continue

                        _writer.add_scalar(f"{get_cfg().model}/val/{k}", v, global_step=engines.global_step)

                if _stopper(stats["val_loss"], save_fn=lambda: engines.save_checkpoint(tag="best")):
                    _logger.warning("Early stop!")
                    early_stop = True

                engines.train()

            if command in ["quit"] or early_stop:
                return
