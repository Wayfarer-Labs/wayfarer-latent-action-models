import  os
import  torch
import  pathlib
from    typing  import TypedDict
from    torch   import nn, Tensor

CKPT_DIR = pathlib.Path('checkpoints/')


class LogStats(TypedDict):
    # TODO Loss, rankme?
    pass


class Trainer_LatentActionModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # TODO Initialize
        self.epoch = 0
        self.ckpt_dir = CKPT_DIR
        if self.should_load: self.load_checkpoint()

    @property
    def should_log(self)    -> bool:
        pass

    @property
    def should_train(self)  -> bool:
        pass

    @property
    def should_save(self)   -> bool:
        pass

    @property
    def should_load(self)   -> bool:
        pass

    @property
    def save_path(self) -> str: return f'lam_e{self.epoch}_s{self.global_step}.pt'

    def load_checkpoint(self, path: os.PathLike) -> None:
        pass

    def save_checkpoint(self, path: os.PathLike) -> None:
        pass

    def format_batch(self) -> Tensor:
        pass

    def compute_loss(self) -> Tensor:
        pass

    def train_step(self, video_bnchw: Tensor) -> dict:
        pass

    def log_step(self, log_stats: dict) -> Tensor:
        pass

    def optim_step(self) -> Tensor:
        pass


    def train_epoch(self)   -> Tensor:
        while self.should_train:
            # TODO Handle epochs, global iter
            video_bnchw: Tensor   = self.format_batch()
            info:        LogStats = self.train_step()
            if self.should_log:     self.log_step(info)
            if self.should_save:    self.save_checkpoint(self.ckpt_dir / self.save_path)
