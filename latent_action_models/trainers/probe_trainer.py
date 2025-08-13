# probes.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops as eo
from typing import Optional, Literal, Any
import wandb

ModelType = Literal["linear", "mlp"]

class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.head = nn.Linear(in_dim, out_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)

class MLPProbe(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 512, depth: int = 2, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.GELU()]
            if dropout > 0: layers.append(nn.Dropout(dropout))
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

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
        lam_model,                      # LatentActionModel (already on correct device)
        probe_cfg,                      # ProbeConfig
        device: torch.device,
        *,
        rank: int = 0,
        model_type: ModelType = "linear",
        wandb_enabled: bool = True,
        wandb_prefix: str = "probe",
        mlp_hidden: int = 512,
        mlp_depth: int = 2,
        mlp_dropout: float = 0.0,
        max_steps: Optional[int] = None,
        per_dim_metrics: bool = False,
    ) -> None:
        self.model = lam_model.eval()
        for p in self.model.parameters(): p.requires_grad = False

        self.cfg           = probe_cfg
        self.device        = device
        self.rank          = rank
        self.model_type    = model_type
        self.wandb_enabled = bool(wandb_enabled and rank == 0)
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
        eval_every: int = 1000,
        project: Optional[str] = None,
        run_name: Optional[str] = None,
    ) -> None:
        if self.rank != 0:
            return
        if self.wandb_enabled and wandb.run is None:
            wandb.init(project=project or "latent-action-models",
                       name=run_name or f"{self.wandb_prefix}_{torch.randint(0,999999,(1,)).item()}")

        step = 0
        for _epoch in range(self.cfg.num_epochs):
            for batch in train_loader:
                step += 1
                loss, metrics = self._train_step(batch)
                if self.wandb_enabled and (step % log_every == 0):
                    wandb.log({f"{self.wandb_prefix}/train_loss": loss,
                               **{f"{self.wandb_prefix}/train_{k}": v for k, v in metrics.items()}},
                              step=step)
                if self.wandb_enabled and val_loader is not None and (step % eval_every == 0):
                    val_metrics = self.evaluate(val_loader)
                    wandb.log({f"{self.wandb_prefix}/val_{k}": v for k, v in val_metrics.items()}, step=step)
                if self.max_steps and step >= self.max_steps:
                    return
        if self.wandb_enabled and val_loader is not None:
            val_metrics = self.evaluate(val_loader)
            wandb.log({f"{self.wandb_prefix}/val_{k}": v for k, v in val_metrics.items()})

    @torch.no_grad()
    def evaluate(self, loader) -> dict:
        if self.rank != 0:
            return {}
        self.probe.eval()
        preds_list, tgts_list = [], []
        for batch in loader:
            X, Y = self._batch_to_xy(batch)
            pred = self.probe(X)
            preds_list.append(pred.cpu())
            tgts_list.append(Y.cpu())
        preds = torch.cat(preds_list, dim=0)
        tgts  = torch.cat(tgts_list, dim=0)
        return self._regression_metrics(preds, tgts)

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

    def _train_step(self, batch) -> tuple[float, dict]:
        X, Y = self._batch_to_xy(batch)  # X: [M,D], Y: [M,K] (float)
        if self.probe is None:
            self._build_probe(in_dim=int(X.shape[-1]), out_dim=int(Y.shape[-1]))

        self.probe.train()
        pred = self.probe(X)
        loss = F.mse_loss(pred, Y)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        with torch.no_grad():
            metrics = self._regression_metrics(pred.detach(), Y)
        return float(loss.item()), metrics

    def _regression_metrics(self, preds: torch.Tensor, tgts: torch.Tensor) -> dict:
        mae  = F.l1_loss(preds, tgts).item()
        rmse = torch.sqrt(F.mse_loss(preds, tgts)).item()
        ybar = tgts.mean(dim=0, keepdim=True)
        ss_tot = torch.sum((tgts - ybar)**2)
        ss_res = torch.sum((tgts - preds)**2)
        r2 = float((1.0 - ss_res / (ss_tot + 1e-8)).item())
        out = {"mae": mae, "rmse": rmse, "r2": r2}
        if self.per_dim:
            mae_k  = torch.mean(torch.abs(preds - tgts), dim=0)  # [K]
            ybar_k = tgts.mean(dim=0, keepdim=True)
            ss_tot_k = torch.sum((tgts - ybar_k)**2, dim=0)
            ss_res_k = torch.sum((tgts - preds )**2, dim=0)
            r2_k = 1.0 - (ss_res_k / (ss_tot_k + 1e-8))
            for i, (m, r) in enumerate(zip(mae_k.tolist(), r2_k.tolist())):
                out[f"mae_k/{i}"] = m
                out[f"r2_k/{i}"]  = r
        return out

    # ------------- dataset alignment -------------
    def _batch_to_xy(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Input batch: (video_bnchw, meta)
        - encodes μ from video using lam_model.encode_to_actions -> [B,N,1,D]
        - extracts targets from meta:
            preferred: 'agg_actions'  -> [B,K] or per-sample [K]
            fallback : 'agg_mouse'+'agg_button' -> concat to [B,K]
            or       : 'actions_bnd' -> [B,N',K] ; 'actions' -> [B,N,K]
        - aligns latents to targets:
            * per-sample Y [B,K] -> X [B,D] by reducing latents over N
            * per-interval Y [B,N',K] -> X [(B·N'),D] by time-aligning then flattening
        Returns:
          X: [M,D], Y: [M,K]
        """
        assert isinstance(batch, (tuple, list)) and len(batch) >= 2, "Expected (video_bnchw, meta)"
        video_bnchw, meta = batch[0], batch[1]
        video_bnchw = video_bnchw.to(self.device, non_blocking=True).float()
        B = video_bnchw.shape[0]

        # Latents: [B,N,1,D] -> [B,N,D]
        with torch.no_grad():
            act = self.model.encode_to_actions(video_bnchw)
            mu_bn1d: torch.Tensor = act["mean_bn1d"]
            mu_bnd = eo.rearrange(mu_bn1d, "b n 1 d -> b n d").contiguous()  # [B,N,D]
            N_lat = mu_bnd.shape[1]

        # Targets
        Y, mode = self._extract_targets(meta, B=B)  # mode: 'sample' or 'interval'

        if mode == "sample":
            # Y: [B,K]
            if N_lat == 1:
                X = mu_bnd[:, 0, :]                              # [B,D]
            else:
                X = mu_bnd.mean(dim=1)                           # [B,D] (reduce across steps)
            return X.to(self.device), Y.to(self.device).float()

        elif mode == "interval":
            # Y: [B,Nt,K] ; align Nt with available latent steps
            Nt = Y.shape[1]
            if N_lat < Nt:
                # truncate targets to available latent steps
                Y = Y[:, :N_lat, :]
                Nt = N_lat
            elif N_lat > Nt:
                # truncate latents to targets
                mu_bnd = mu_bnd[:, :Nt, :]

            X = eo.rearrange(mu_bnd, "b n d -> (b n) d")         # [(B*Nt), D]
            Y = eo.rearrange(Y,      "b n k -> (b n) k")         # [(B*Nt), K]
            return X.to(self.device), Y.to(self.device).float()

        else:
            raise RuntimeError(f"Unknown target alignment mode: {mode}")

    def _extract_targets(self, meta: Any, B: int) -> tuple[torch.Tensor, str]:
        """
        Returns (targets, mode), where mode ∈ {'sample','interval'}.
        Accepts:
          - dict meta (batched) or list/tuple of per-sample dicts (un-collated).
          - preferred: 'agg_actions' -> per-sample [B,K] (or [K] pre-collate)
          - optional:  concat('agg_mouse','agg_button') -> [B,K]
          - fallback:  'actions_bnd' -> [B,N-1,K] (interval), or 'actions' -> [B,N,K] (instantaneous)
        """
        def to_tensor(x):
            return x if isinstance(x, torch.Tensor) else torch.tensor(x)

        # (A) batched dict
        if isinstance(meta, dict):
            if "agg_actions" in meta:
                y = to_tensor(meta["agg_actions"])
                if y.ndim == 1: y = y.unsqueeze(0)                # [K] -> [1,K]
                return y, "sample"
            if "agg_mouse" in meta and "agg_button" in meta:
                y = torch.cat([to_tensor(meta["agg_mouse"]), to_tensor(meta["agg_button"])], dim=-1)
                if y.ndim == 1: y = y.unsqueeze(0)
                return y, "sample"
            if "actions_bnd" in meta:
                y = to_tensor(meta["actions_bnd"])                # [B,N-1,K] or [N-1,K]
                if y.ndim == 2: y = y.unsqueeze(0)                # -> [1,N-1,K]
                return y, "interval"
            if "actions" in meta:
                y = to_tensor(meta["actions"])                    # [B,N,K] or [N,K]
                if y.ndim == 2: y = y.unsqueeze(0)
                return y, "interval"

        # (B) list/tuple of per-sample dicts (your latent_collate_fn style)
        if isinstance(meta, (list, tuple)) and len(meta) == B and isinstance(meta[0], dict):
            # try preferred key first
            if "agg_actions" in meta[0]:
                y = torch.stack([to_tensor(m["agg_actions"]) for m in meta], dim=0)  # [B,K]
                return y, "sample"
            if ("agg_mouse" in meta[0]) and ("agg_button" in meta[0]):
                y = torch.stack([torch.cat([to_tensor(m["agg_mouse"]), to_tensor(m["agg_button"])], dim=-1)
                                 for m in meta], dim=0)                                # [B,K]
                return y, "sample"
            if "actions_bnd" in meta[0]:
                y = torch.stack([to_tensor(m["actions_bnd"]) for m in meta], dim=0)   # [B,N-1,K]
                return y, "interval"
            if "actions" in meta[0]:
                y = torch.stack([to_tensor(m["actions"]) for m in meta], dim=0)       # [B,N,K]
                return y, "interval"

        raise KeyError(
            "Probe targets not found in meta. Expected one of: "
            "'agg_actions' (preferred), 'agg_mouse'+'agg_button', 'actions_bnd', or 'actions'."
        )
