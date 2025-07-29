import  math
import  toolz
import  torch
import  torch.nn as nn
from    einops  import rearrange
from    torch   import Tensor

from    latent_action_models.models.rotary import RotaryEmbedding


class PositionalEncoding(nn.Module):
    def __init__(self,
                 model_dim: int,
                 max_len: int = 5000) -> None:

        super(PositionalEncoding, self).__init__()
        pe          = torch.zeros(max_len, model_dim)
        position    = torch.arange(0, max_len)      .float().unsqueeze(1)
        exponent    = torch.arange(0, model_dim, 2) .float() * -(math.log(10000.0) / model_dim)
        pe[:, 0::2] = torch.sin(position).mul(torch.exp(exponent))
        pe[:, 1::2] = torch.cos(position).mul(torch.exp(exponent))
        self.pos_enc = pe

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pos_enc[:x.shape[2]].to(x.device)

class CrossAttention(nn.Module):
    def __init__(
        self,
        num_heads:  int,
        d_model:    int,
        dropout:    float   = 0.0,
        qkv_bias:   bool    = False,
        proj_bias:  bool    = True,
        qk_norm:    bool    = True,
        attn_drop:  float   = 0.0,
        use_rotary: bool    = False,
    ):
        super().__init__()

        self.num_heads  = num_heads
        self.head_dim   = d_model // num_heads

        # Scaling by 8 to be equal when head_dim=64
        self.scale      = math.sqrt(self.head_dim)
        self.q          = nn.Linear(d_model, d_model * 1, bias=qkv_bias)
        self.kv         = nn.Linear(d_model, d_model * 2, bias=qkv_bias)
        self.attn_drop  = nn.Dropout(attn_drop)
        self.proj       = nn.Sequential(
            nn.Linear(d_model, d_model, bias=proj_bias),
            nn.Dropout(dropout)
        )

        # qk normalization https://arxiv.org/pdf/2302.05442
        # Note that LN is done in fp32, so they have to be
        self.qk_norm    = qk_norm      and nn.LayerNorm(self.head_dim, eps=1e-05)
        self.rotary     = use_rotary   and RotaryEmbedding(dim=self.head_dim)

    def forward(self, x_query: torch.Tensor, x_context: torch.Tensor) -> torch.Tensor:
        B, N_q, Cq = x_query  .shape
        B, N_c, _ = x_context.shape

        q   = self.q(x_query)
        q   = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)

        kv  = self.kv(x_context)
        k,v = rearrange(kv, 'b n (two h d) -> two b h n d', two=2, h=self.num_heads)

        if self.rotary:
            q = self.rotary.rotate_queries_or_keys(q, self.rotary.freqs).contiguous() 
            k = self.rotary.rotate_queries_or_keys(k, self.rotary.freqs).contiguous()

        if self.qk_norm:
            q,k = self.qk_norm(q),      self.qk_norm(k)
            q,k = q.to(dtype=v.dtype),  k.to(dtype=v.dtype)

        q *= self.scale

        attn: Tensor    = q @ k.transpose(-2, -1)
        attn            = attn.softmax(dim=-1)
        attn            = self.attn_drop(attn)

        x = rearrange(attn @ v, 'b h n d -> b n (h d)')
        x = self.proj(x)
        return x


class SelfAttention(nn.Module):
    def __init__(
        self,
        num_heads:  int,
        d_model:    int,
        dropout:    float   = 0.0,
        causal:     bool    = True,
        qkv_bias:   bool    = False,
        proj_bias:  bool    = True,
        qk_norm:    bool    = True,
        attn_drop:  float   = 0.0,
        use_rotary: bool    = False,
    ) -> None:
        super().__init__()

        self.num_heads  = num_heads
        self.head_dim   = d_model // num_heads

        # Scaling by 8 to be equal when head_dim=64
        self.causal     = causal
        self.scale      = math.sqrt(self.head_dim)
        self.qkv        = nn.Linear(d_model, d_model * 3, bias=qkv_bias)
        self.attn_drop  = nn.Dropout(attn_drop)
        self.proj       = nn.Sequential(
            nn.Linear(d_model, d_model, bias=proj_bias),
            nn.Dropout(dropout)
        )

        # qk normalization https://arxiv.org/pdf/2302.05442
        # Note that LN is done in fp32, so they have to be
        self.qk_norm    = qk_norm      and nn.LayerNorm(self.head_dim, eps=1e-05)
        self.rotary     = use_rotary   and RotaryEmbedding(dim=self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        qkv: tuple[Tensor, ...] = rearrange(
            self.qkv(x), 
            'b n (three h d) -> three b h n d', 
            three=3, h=self.num_heads, d=C // self.num_heads
        )

        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.rotary:
            q = self.rotary.rotate_queries_or_keys(q, self.rotary.freqs).contiguous() 
            k = self.rotary.rotate_queries_or_keys(k, self.rotary.freqs).contiguous()

        if self.qk_norm:
            q = self.qk_norm(q)
            k = self.qk_norm(k)
            # LN done in float32, cast back to bf16
            q = q.to(dtype=v.dtype)
            k = k.to(dtype=v.dtype)
        
        q *= self.scale

        attn: Tensor = q @ k.transpose(-2, -1)

        if self.causal:
            mask_value = -torch.finfo(attn.dtype).max
            i, j = attn.shape[-2:]            
            mask = ~torch.tril(torch.ones(i, j)).bool().to(attn.device)
            attn = attn.masked_fill(mask, mask_value)

        attn = attn.softmax(dim=-1)
        x = rearrange(attn @ v, 'b h n d -> b n (h d)') ; del q,k,v
        x = self.proj(x)
        return x


class MLP(nn.Module):
    def __init__(self, d_model: int, ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(d_model * ratio)
        self.fc1 = nn.Linear(d_model, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class ST_Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        attn_drop: float = 0.0,
        mlp_ratio: float = 4.0,
        mlp_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm_s = nn.LayerNorm(d_model)
        self.norm_t = nn.LayerNorm(d_model)
        self.s_attn = SelfAttention(d_model=d_model, num_heads=num_heads, attn_drop=attn_drop,
                                    use_rotary=False, causal=False)
        self.t_attn = SelfAttention(d_model=d_model, num_heads=num_heads, attn_drop=attn_drop,
                                    use_rotary=True, causal=False)
        self.norm_m = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, ratio=mlp_ratio, drop=mlp_drop)

    def forward(self, x: Tensor) -> Tensor:  # x: [B,T,S,C]
        B, T, S, C = x.shape
        # spatial
        xs = rearrange(x, 'B T S C -> (B T) S C')
        xs = xs + self.s_attn(self.norm_s(xs))
        x = rearrange(xs, '(B T) S C -> B T S C', B=B)
        # temporal (bidirectional)
        xt = rearrange(x, 'B T S C -> (B S) T C')
        xt = xt + self.t_attn(self.norm_t(xt))
        x = rearrange(xt, '(B S) T C -> B T S C', S=S)
        # MLP
        x = x + self.mlp(self.norm_m(x))
        return x


class S_Block(nn.Module):
    def __init__(
        self,
        d_model:    int,
        num_heads:  int,
        attn_drop:  float = 0.0,
        mlp_ratio:  float = 4.0,
        mlp_drop:   float = 0.0,
    ) -> None:
        super().__init__()
        self.norm_s = nn.LayerNorm(d_model)
        self.s_attn = SelfAttention(d_model=d_model, num_heads=num_heads, attn_drop=attn_drop,
                                    use_rotary=False, causal=False)
        self.norm_m = nn.LayerNorm(d_model)
        self.mlp    = MLP(d_model, ratio=mlp_ratio, drop=mlp_drop)

    def forward(self, in_bnsc: Tensor) -> Tensor:
        B, N, S, C = in_bnsc.shape
        # spatial
        x_bsc   = rearrange(in_bnsc, 'B N S C -> (B N) S C')
        x_bsc   = x_bsc + self.s_attn(self.norm_s(x_bsc))
        x_bnsc  = rearrange(x_bsc, '(B N) S C -> B N S C', B=B)
        # MLP
        x_bnsc  = x_bnsc + self.mlp(self.norm_m(x_bnsc))
        return x_bnsc


class Block_Transformer(nn.Module):
    def __init__(self,
            in_dim:     int,
            model_dim:  int,
            out_dim:    int,
            num_blocks: int,
            num_heads:  int,
            dropout:    float    = 0.0,
            blk_kwargs: dict     = dict(),
            blk_class:  type     = ST_Block
        ):
        super(Block_Transformer, self).__init__()
        self.in_proj    = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear   (in_dim, model_dim),
            nn.LayerNorm(model_dim)
        )
        self.pos_embed  = PositionalEncoding(model_dim)
        self.blocks     = nn.ModuleList([
            blk_class(model_dim, num_heads, **blk_kwargs)
            for _ in range(num_blocks)
        ])
        self.out_proj   = nn.Linear(model_dim, out_dim)

    def forward(self, x_bnpd: Tensor) -> Tensor:
        x = self.in_proj  (x_bnpd)
        x = self.pos_embed(x)
        x = toolz.pipe    (x, *self.blocks)
        x = self.out_proj (x)
        return x


class ST_Transformer(Block_Transformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, blk_class=ST_Block)


class S_Transformer(Block_Transformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, blk_class=S_Block)
