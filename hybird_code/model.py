import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from functools import lru_cache
from operator import mul
from functools import reduce
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_
from interview_feature_exchange import InterviewFeatureExchange


# --------------------------------------------------
# MLP
# --------------------------------------------------

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


# --------------------------------------------------
# Window utilities
# --------------------------------------------------

def window_partition(x, window_size):
    """
    x: (B, D, H, W, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(
        B,
        D // window_size[0], window_size[0],
        H // window_size[1], window_size[1],
        W // window_size[2], window_size[2],
        C
    )
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    return windows.view(-1, reduce(mul, window_size), C)


def window_reverse(windows, window_size, B, D, H, W):
    x = windows.view(
        B,
        D // window_size[0],
        H // window_size[1],
        W // window_size[2],
        window_size[0], window_size[1], window_size[2],
        -1
    )
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    return x.view(B, D, H, W, -1)


def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    use_shift_size = list(shift_size) if shift_size is not None else None

    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    return (tuple(use_window_size), tuple(use_shift_size)) if shift_size else tuple(use_window_size)


# --------------------------------------------------
# Window Attention 3D
# --------------------------------------------------

class WindowAttention3D(nn.Module):
    def __init__(self, dim, window_size, num_heads,
                 qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1) *
                (2 * window_size[1] - 1) *
                (2 * window_size[2] - 1),
                num_heads
            )
        )

        coords = torch.stack(
            torch.meshgrid(
                torch.arange(window_size[0]),
                torch.arange(window_size[1]),
                torch.arange(window_size[2]),
                indexing="ij"
            )
        )
        coords_flatten = coords.flatten(1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()

        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 2] += window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)

        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index[:N-1, :N-1].reshape(-1)
        ].reshape(N-1, N-1, -1).permute(2, 0, 1)

        attn[:, :, 1:, 1:] += relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn[:, :, 1:, 1:] = attn[:, :, 1:, 1:].view(
                B_ // nW, nW, self.num_heads, N-1, N-1
            ) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj_drop(self.proj(x))

        return x[:, 1:], x[:, :1]


# --------------------------------------------------
# intraview_feature_exchange
# --------------------------------------------------

class intraview_feature_exchange(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, use_checkpoint=False):

        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.use_checkpoint = use_checkpoint

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim, window_size, num_heads,
            qkv_bias, qk_scale, attn_drop, drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x, global_token, position, attn_mask):
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)

        shortcut = x
        x = self.norm1(x)

        pad = (
            0, 0,
            0, (window_size[2] - W % window_size[2]) % window_size[2],
            0, (window_size[1] - H % window_size[1]) % window_size[1],
            0, (window_size[0] - D % window_size[0]) % window_size[0],
        )
        x = F.pad(x, pad)
        _, Dp, Hp, Wp, _ = x.shape

        if any(shift_size):
            x = torch.roll(x, (-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))

        x_windows = window_partition(x, window_size)

        if global_token is None:
            global_token = position.to(x.device)

        x_windows = torch.cat([global_token, x_windows], dim=1)
        x_windows, global_token = self.attn(x_windows, attn_mask)

        x = window_reverse(x_windows, window_size, B, Dp, Hp, Wp)

        if any(shift_size):
            x = torch.roll(x, shift_size, dims=(1, 2, 3))

        x = x[:, :D, :H, :W, :]
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        global_token = global_token + self.drop_path(self.mlp(self.norm2(global_token)))
        return x, global_token


# --------------------------------------------------
# Mask computation
# --------------------------------------------------

@lru_cache()
def compute_mask(D, H, W, window_size, shift_size, device):
    img_mask = torch.zeros((1, D, H, W, 1), device=device)
    cnt = 0
    for d in (slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None)):
        for h in (slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None)):
            for w in (slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None)):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1

    mask_windows = window_partition(img_mask, window_size).squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    return attn_mask.masked_fill(attn_mask != 0, -100.).masked_fill(attn_mask == 0, 0.)


# --------------------------------------------------
# Patch embedding
# --------------------------------------------------

class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size=(2, 4, 4), in_chans=1, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        _, _, D, H, W = x.shape
        pad = (
            0, (self.patch_size[2] - W % self.patch_size[2]) % self.patch_size[2],
            0, (self.patch_size[1] - H % self.patch_size[1]) % self.patch_size[1],
            0, (self.patch_size[0] - D % self.patch_size[0]) % self.patch_size[0],
        )
        x = F.pad(x, pad)
        x = self.proj(x)

        if self.norm:
            B, C, D, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(B, C, D, H, W)

        return x

# --------------------------------------------------
# BasicLayer
# --------------------------------------------------

class BasicLayer(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=(6, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            intraview_feature_exchange(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
            )
            for i in range(depth)])

    def forward(self, x, global_token, position):
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)
        for blk in self.blocks:
            x, global_token = blk(x, global_token, position, attn_mask)
        x = x.view(B, D, H, W, -1)
        global_token = global_token.view(B, -1, C)

        return x, global_token

class PatchMerging(nn.Module):
    """ Patch Merging Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, D, H, W, C).
        """
        B, D, H, W, C = x.shape

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, :, 0::2, 0::2, :]  # B D H/2 W/2 C
        x1 = x[:, :, 1::2, 0::2, :]  # B D H/2 W/2 C
        x2 = x[:, :, 0::2, 1::2, :]  # B D H/2 W/2 C
        x3 = x[:, :, 1::2, 1::2, :]  # B D H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B D H/2 W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

class DownsampleAndRearrange(nn.Module):
    def __init__(self, downsample, window_size, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.window_size = window_size
        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)

    def forward(self, x, global_token):
        B, D, H, W, _ = x.shape
        C = global_token.shape[-1]
        global_token = global_token.view(-1, 1, C)

        if self.downsample is not None:
            x = self.downsample(x)
            global_token = global_token.view(B, D//self.window_size[0], H // self.window_size[1], W // self.window_size[2], C)
            global_token = self.downsample(global_token)
            global_token = global_token.view(-1, 1, 2 * C)
        x = rearrange(x, 'b d h w c -> b c d h w')

        return x, global_token

class Hybirdfusion(nn.Module):
    """
    Swin Transformer 3D backbone with multi-view global token interaction.
    This implementation is behavior-equivalent to the original version,
    with only debugging / visualization code removed.
    """

    def __init__(self,
                 pretrained=None,
                 pretrained2d=True,
                 patch_size=(1, 4, 4),
                 in_chans=1,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=(6, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=False,
                 frozen_stages=-1,
                 use_checkpoint=False,
                 num_classes=2):
        super().__init__()

        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.window_size = window_size
        self.patch_size = patch_size
        self.num_classes = num_classes

        # Patch embedding
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # backbone layers: [BasicLayer, DownsampleAndRearrange, VIT] × stages
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            dim = int(embed_dim * 2 ** i_layer)

            self.layers.append(
                BasicLayer(
                    dim=dim,
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    norm_layer=norm_layer,
                    use_checkpoint=use_checkpoint
                )
            )

            self.layers.append(
                DownsampleAndRearrange(
                    downsample=PatchMerging if i_layer < self.num_layers - 1 else None,
                    window_size=window_size,
                    dim=dim,
                    norm_layer=norm_layer
                )
            )

            self.layers.append(
                InterviewFeatureExchange(
                    embed_dim=dim,
                    num_heads=12,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                )
            )

        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

        # heads
        self.norm = norm_layer(self.num_features)
        self.avgpool3d = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.conv = nn.Conv3d(self.num_features * 3, self.num_features, 1)
        self.fc = nn.Linear(self.num_features, self.num_classes)

        # global positional tokens
        self.global_position = nn.Parameter(torch.zeros(192, 1, embed_dim))
        self.global_position1 = nn.Parameter(torch.zeros(192, 1, embed_dim))
        self.global_position2 = nn.Parameter(torch.zeros(192, 1, embed_dim))

        self._freeze_stages()

    # ------------------------------------------------------------------
    # utility
    # ------------------------------------------------------------------

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for p in self.patch_embed.parameters():
                p.requires_grad = False

        if self.frozen_stages >= 1:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages):
                m = self.layers[i]
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    def compute_window_centers_in_patch(self, x, affine1, affine2):
        """
        Compute window center correspondence under affine transformation.
        This function is kept identical to the original implementation.
        """
        B, C, D, H, W = x.shape
        affine1 = np.squeeze(affine1)
        affine2 = np.squeeze(affine2)

        z_indices = (np.arange(0, D, self.window_size[0]) + self.window_size[0] // 2)
        y_indices = (np.arange(0, H, self.window_size[1]) + self.window_size[1] // 2) * 4 + 2
        x_indices = (np.arange(0, W, self.window_size[2]) + self.window_size[2] // 2) * 4 + 2

        z_grid, y_grid, x_grid = np.meshgrid(z_indices, y_indices, x_indices, indexing='ij')
        start_coords = np.stack([y_grid, x_grid, z_grid], axis=-1).reshape(-1, 3)

        start_coords_h = np.concatenate(
            [start_coords, np.ones((start_coords.shape[0], 1))], axis=1
        )

        position = np.dot(start_coords_h, affine1.T)
        inv_affine2 = np.linalg.inv(affine2)
        position = np.dot(position, inv_affine2.T)[:, :3]

        valid = (
            (0 <= position[:, 0]) & (position[:, 0] < 224) &
            (0 <= position[:, 1]) & (position[:, 1] < 224) &
            (0 <= position[:, 2]) & (position[:, 2] < 18)
        )

        result = np.zeros_like(self.global_position.detach().cpu().numpy())
        for i in range(position.shape[0]):
            if valid[i]:
                dist = np.linalg.norm(start_coords - position[i], axis=1)
                idx = np.argmin(dist)
                result[i] = self.global_position[idx].detach().cpu().numpy()

        return result

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, x1, x2, x3, affine1, affine2, affine3):
        """
        x1, x2, x3: axial / coronal / sagittal views
        """

        x1 = self.patch_embed(x1)
        x2 = self.patch_embed(x2)
        x3 = self.patch_embed(x3)

        position1 = self.global_position
        position2 = torch.from_numpy(
            self.compute_window_centers_in_patch(x2, affine2, affine1)
        ).to(x1.device)
        position3 = torch.from_numpy(
            self.compute_window_centers_in_patch(x3, affine3, affine1)
        ).to(x1.device)

        global_token1 = None
        global_token2 = None
        global_token3 = None

        x1 = self.pos_drop(x1)
        x2 = self.pos_drop(x2)
        x3 = self.pos_drop(x3)

        for i in range(0, len(self.layers), 3):
            layer = self.layers[i]
            downsample = self.layers[i + 1]
            vit = self.layers[i + 2]

            x1, global_token1 = layer(x1.contiguous(), global_token1, position1)
            x2, global_token2 = layer(x2.contiguous(), global_token2, position2)
            x3, global_token3 = layer(x3.contiguous(), global_token3, position3)

            global_token1, global_token2, global_token3 = vit(
                global_token1, global_token2, global_token3
            )

            x1, global_token1 = downsample(x1, global_token1)
            x2, global_token2 = downsample(x2, global_token2)
            x3, global_token3 = downsample(x3, global_token3)

        # normalization
        x1 = self.norm(rearrange(x1, 'n c d h w -> n d h w c'))
        x2 = self.norm(rearrange(x2, 'n c d h w -> n d h w c'))
        x3 = self.norm(rearrange(x3, 'n c d h w -> n d h w c'))

        x1 = rearrange(x1, 'n d h w c -> n c d h w')
        x2 = rearrange(x2, 'n d h w c -> n c d h w')
        x3 = rearrange(x3, 'n d h w c -> n c d h w')

        y = torch.cat((x1, x2, x3), dim=1)
        y = self.conv(y)
        y = self.avgpool3d(y)
        y = torch.flatten(y, 1)

        return self.fc(y)

def test():
    model = Hybirdfusion()
    x = torch.randn(1,1,18,224,224)
    affine = torch.eye(4).unsqueeze(0)
    y = model(x, x, x, affine, affine, affine)
    print(y.shape)


if __name__ == "__main__":
    test()