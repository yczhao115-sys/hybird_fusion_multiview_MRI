import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_


# -----------------------------
# Attention
# -----------------------------
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# -----------------------------
# MLP
# -----------------------------
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.0, act_layer=nn.GELU):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# -----------------------------
# Transformer Block
# -----------------------------
class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=drop,
            act_layer=act_layer,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ============================================================
# Interview Feature Exchange Transformer
# ============================================================
class InterviewFeatureExchange(nn.Module):

    def __init__(
        self,
        embed_dim=256,
        depth=2,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.1,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
    ):
        super().__init__()

        dpr = torch.linspace(0, drop_path, depth).tolist()

        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x1, x2, x3):
        len_x1, len_x2, len_x3 = x1.size(1), x2.size(1), x3.size(1)

        # feature exchange
        y = torch.cat([x1, x2, x3], dim=1)

        y = self.blocks(y)
        y = self.norm(y)

        # split back
        x1 = y[:, :len_x1]
        x2 = y[:, len_x1 : len_x1 + len_x2]
        x3 = y[:, len_x1 + len_x2 :]

        return x1, x2, x3


# -----------------------------
# Minimal Test
# -----------------------------
if __name__ == "__main__":
    B, C = 1, 96
    x1 = torch.randn(B, 64, C)
    x2 = torch.randn(B, 32, C)
    x3 = torch.randn(B, 48, C)

    model = InterviewFeatureExchange(embed_dim=C)
    y1, y2, y3 = model(x1, x2, x3)

    print(y1.shape, y2.shape, y3.shape)
