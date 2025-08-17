# probes.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops as eo
from typing import Optional, Literal, Any
import wandb
from latent_action_models.configs import ProbeConfig
from latent_action_models.models.latent_action_model import LatentActionModel

ModelType = Literal["linear", "mlp"]

class RegressionProbeTrainer:
    """
    Rank-0-only trainer mapping latent actions μ -> ground-truth action vector (regression).
    Works with your dataset's keys:
      - preferred: 'agg_actions'       (shape [B,K] after collate; or per-sample [K] before collate)
      - optional:  'agg_mouse'+'agg_button' (concatenated)
      - fallback:  'actions_bnd' ([B,N-1,K]) or 'actions' ([B,N,K]) for per-interval/instantaneous
    Aligns latents to targets automatically.
    """
    def __init__(
        self,
        lam_model: LatentActionModel,                      # LatentActionModel (already on correct device)
        probe_cfg: ProbeConfig,                      # ProbeConfig
        device: torch.device,
        *,
        rank: int = 0,
        model_type: ModelType = "linear",
        wandb_prefix: str = "probe",
        mlp_hidden: int = 512,
        mlp_depth: int = 2,
        mlp_dropout: float = 0.0,
        max_steps: Optional[int] = None,
        per_dim_metrics: bool = False,
    ) -> None:
        self.model = lam_model.eval()

        self.cfg           = probe_cfg
        self.device        = device
        self.rank          = rank
        self.model_type    = model_type
        self.wandb_prefix  = f"{wandb_prefix}/{model_type}"
        self.mlp_hidden    = mlp_hidden
        self.mlp_depth     = mlp_depth
        self.mlp_dropout   = mlp_dropout
        self.max_steps     = max_steps
        self.per_dim       = per_dim_metrics

        self.probe: Optional[nn.Module] = None
        self.opt:   Optional[torch.optim.Optimizer] = None



    # ---------------- public API ----------------
    def fit(
        self,
        train_loader,
        val_loader: Optional[Any] = None,
        *,
        log_every: int = 100,
        at_step: int = 0,
    ) -> None:
        if self.rank != 0:
            return    

        iter_loader = iter(train_loader)
        step = 0
        for _epoch in range(self.cfg.num_epochs):
            epoch_step = 0
            for batch in iter_loader:

                if epoch_step > self.cfg.samples_per_epoch:
                    break
                
                loss, metrics = self._train_step(batch)
                
                if (step % log_every == 0):
                    wandb.log({
                        "global_step": at_step,  # align to trainer’s step
                        f"{self.wandb_prefix}/train_loss": loss,
                        **{f"{self.wandb_prefix}/train_{k}": v for k, v in metrics.items()},
                    })

                if self.max_steps and step >= self.max_steps:
                    return
                
                step += 1 ; epoch_step += 1

        if val_loader is not None:
            val_metrics = self.evaluate(val_loader)
            wandb.log({
                "global_step": at_step,
                **{f"{self.wandb_prefix}/val_{k}": v for k, v in val_metrics.items()},
            })

    @torch.no_grad()
    def evaluate(self, loader) -> dict:
        try:
            if self.rank != 0:
                return {}
            self.probe.eval()
            iter_loader = iter(loader)
            preds_list, tgts_list = [], []
            step = 0
            for batch in iter_loader:
                X, Y = self._batch_to_xy(batch)
                pred = self.probe(X)
                preds_list.append(pred.cpu())
                tgts_list.append(Y.cpu())
                if self.max_steps and step >= self.max_steps:
                    break
                step += 1
            preds = torch.cat(preds_list, dim=0)
            tgts  = torch.cat(tgts_list, dim=0)
            return self._regression_metrics(preds, tgts)
        finally:
            self.probe.train()

    # ---------------- internals ----------------
    def _build_probe(self, in_dim: int, out_dim: int):
        if self.model_type == "linear":
            self.probe = LinearProbe(in_dim, out_dim).to(self.device)
        elif self.model_type == "mlp":
            self.probe = MLPProbe(in_dim, out_dim,
                                  hidden=self.mlp_hidden,
                                  depth=self.mlp_depth,
                                  dropout=self.mlp_dropout).to(self.device)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        self.opt = torch.optim.Adam(self.probe.parameters(), lr=self.cfg.lr)


    # ------------- dataset alignment -------------
