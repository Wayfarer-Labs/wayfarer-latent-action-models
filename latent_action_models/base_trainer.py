import pathlib, torch, time
from contextlib        import nullcontext
from torch.optim       import AdamW
from torch.nn          import utils
from torch             import nn, Tensor

from latent_action_models.configs   import BaseTrainerConfig
from latent_action_models.utils     import init_distributed


class BaseTrainer(nn.Module):
    def __init__(self, model: nn.Module, cfg: BaseTrainerConfig,
                 device: torch.device | str = "cuda"):
        super().__init__()
        rank, world_size, device = init_distributed()
        self.rank, self.world_size, self.device = rank, world_size, device
        
        self.model        = model.to(device)
        self.cfg          = cfg
        self.device       = torch.device(device)
        self.global_step  = 0

        # -- dirs
        self.ckpt_dir = pathlib.Path(cfg.ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # -- optimisation
        self.optimizer      = AdamW(self.model.parameters(),
                                lr=cfg.lr,
                                weight_decay=cfg.weight_decay,
                                betas=cfg.betas)

        # -- TODO choose a real schedule lol
        self.scheduler      = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                            cfg.lr_lambda)
        self.max_grad_norm  = cfg.max_grad_norm
        self.use_amp        = cfg.amp
        self.scaler         = torch.amp.GradScaler(self.device.type)

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

    def amp_ctx(self):
        return torch.amp.autocast(self.device.type) if self.use_amp else nullcontext()
