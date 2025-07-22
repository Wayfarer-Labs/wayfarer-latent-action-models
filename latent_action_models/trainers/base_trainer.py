import pathlib, torch, time, os, abc
from abc import abstractmethod
from contextlib        import nullcontext
from torch.optim       import AdamW
from torch.nn          import utils
from torch             import nn, Tensor

from latent_action_models.configs           import BaseTrainerConfig
from latent_action_models.utils             import init_distributed
from latent_action_models.data.dataloaders  import create_dataloader

CKPT_DIR = 'checkpoints/' # global cause im very cool and special

class BaseTrainer(nn.Module):
    def __init__(self, model: nn.Module, cfg: BaseTrainerConfig,
                 device: torch.device | str = "cuda"):
        super().__init__()
        rank, world_size, device = init_distributed()
        self.rank, self.world_size, self.device = rank, world_size, device
        
        self.model          = model.to(device)
        self.cfg            = cfg
        self.device         = torch.device(device)
        self.global_step    = 0
        self.batch_size     = cfg.data_config.batch_size
        # -- data
        self.dataloader     = create_dataloader(self.cfg.data_config)
        self.iter_loader    = iter(self.dataloader)
        # -- dirs
        self.ckpt_dir       = pathlib.Path(cfg.ckpt_dir or CKPT_DIR)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # -- optimisation
        self.optimizer      = AdamW(self.model.parameters(),
                                lr=cfg.lr,
                                weight_decay=cfg.weight_decay,
                                betas=cfg.betas)

        # -- TODO choose a real schedule lol
        self.scheduler      = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, cfg.max_steps)
        self.max_grad_norm  = cfg.max_grad_norm
        self.use_amp        = cfg.amp
        self.scaler         = torch.amp.GradScaler(self.device.type)

        if self.world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.device.index],
                output_device=self.device.index,
                find_unused_parameters=False)
        
        if self.should_load: self.load_checkpoint(self.cfg.resume_checkpoint)

    @abstractmethod
    def load_checkpoint(self, path: os.PathLike) -> None: raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self, path: os.PathLike) -> None: raise NotImplementedError

    def optim_step(self, loss: Tensor) -> float:
        self.optimizer.zero_grad(set_to_none=True)

        if self.use_amp:
            self.scaler.scale(loss).backward()
            if self.max_grad_norm:
                self.scaler.unscale_(self.optimizer)
                grad_norm = utils.clip_grad_norm_(self.model.parameters(),
                                                  self.max_grad_norm)
            else:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.tensor(0., device=loss.device)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            grad_norm = utils.clip_grad_norm_(self.model.parameters(),
                                              self.max_grad_norm) \
                        if self.max_grad_norm else torch.tensor(0.,
                                                                 device=loss.device)
            self.optimizer.step()

        self.scheduler.step()
        return float(grad_norm)

    def _model_unwrapped(self) -> nn.Module:
        if isinstance(self.model, nn.parallel.DistributedDataParallel): return self.model.module
        return self.model

    @property
    def should_log(self) -> bool:
        return self.global_step % self.cfg.log_every == 0   and self.rank == 0

    @property
    def should_save(self) -> bool:
        return self.global_step % self.cfg.ckpt_every == 0  and self.rank == 0

    @property
    def should_load(self) -> bool:
        return self.cfg.resume_checkpoint is not None       and self.rank == 0
    
    @property
    def should_train(self) -> bool:
        return self.global_step < self.cfg.max_steps

    @property
    def should_validate(self) -> bool:
        return self.global_step % self.cfg.val_every == 0 and self.rank == 0

    def amp_ctx(self):
        return torch.amp.autocast(self.device.type) if self.use_amp else nullcontext()

    @abstractmethod
    def train(self) -> None:    raise NotImplementedError

    @abstractmethod
    def validate(self) -> None: raise NotImplementedError