
import math

from mindspore.ops import operations as P
from mindspore import ops

import mindspore
from mindspore import nn, Tensor, ParameterTuple
from mindspore.common.initializer import Normal
from mindspore.common import initializer


import numpy as np
import mindspore.numpy as mind_np

from functools import partial

from src.models.blocks.vision_transformer import Block as MindBlock
from src.models.blocks.helpers import to_2tuple
from src.models.blocks.weight_init import KaimingNormal, TruncNormal
from src.models.blocks.misc import Identity


class Mlp(nn.Cell):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp,self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_features, hidden_features, has_bias=True)
        self.act = act_layer()
        self.fc2 = nn.Dense(hidden_features, out_features, has_bias=True)
        self.drop = nn.Dropout(keep_prob = 1.0 - drop)

    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GroupAttention(nn.Cell):
    """
    LSA: self attention within a group
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., ws=1):
        assert ws != 1
        super(GroupAttention, self).__init__()

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or Tensor(head_dim ** -0.5,  mindspore.float32)

        self.q = nn.Dense(in_channels=dim, out_channels=dim, has_bias=qkv_bias)
        self.k = nn.Dense(in_channels=dim, out_channels=dim, has_bias=qkv_bias)
        self.v = nn.Dense(in_channels=dim, out_channels=dim, has_bias=qkv_bias)

        self.attn_drop = nn.Dropout(keep_prob = 1.0 - attn_drop)
        self.proj = nn.Dense(dim, dim, has_bias= True)
        self.proj_drop = nn.Dropout(keep_prob = 1.0 - proj_drop)
        self.ws = ws

    def construct(self, x, H, W):
        B, N, C = x.shape
        h_group, w_group = H // self.ws, W // self.ws

        total_groups = h_group * w_group

        x = ops.Reshape()(x, (B, h_group, self.ws, w_group, self.ws, C))
        x = ops.Transpose()(x, (0, 1, 3, 2, 4, 5))

        # # B, h*w, ws*ws, n_head, head_dim -> B, h * w, n_head, ws * ws, head_dim
        q = ops.Transpose()(ops.Reshape()(self.q(x), (B, total_groups, -1, self.num_heads, C // self.num_heads)), (0, 1, 3, 2, 4))
        k = ops.Transpose()(ops.Reshape()(self.k(x), (B, total_groups, -1, self.num_heads, C // self.num_heads)), (0, 1, 3, 2, 4))
        v = ops.Transpose()(ops.Reshape()(self.v(x), (B, total_groups, -1, self.num_heads, C // self.num_heads)), (0, 1, 3, 2, 4))

        attn = ops.Mul()(ops.BatchMatMul(transpose_b = True)(q, k), self.scale)  # B, h * w, n_head, ws*ws, ws*ws

        attn = nn.Softmax(axis=-1)(attn)
        attn = self.attn_drop(attn) 

        # attn @ v-> B, hw, n_head, ws*ws, head_dim -> (transpose(2,3)) B, h * w, ws*ws, n_head,  head_dim
        attn = ops.Transpose()(ops.BatchMatMul()(attn, v), (0, 1, 3, 2, 4))
        attn = ops.Reshape()(attn, (B, h_group, w_group, self.ws, self.ws, C) )
        x = ops.Reshape()(ops.Transpose()(attn, (0, 1, 3, 2, 4, 5)), (B, N, C))  # B, h , ws, w, ws, C -> B, N, C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention(nn.Cell):
    """
    GSA: using a  key to summarize the information for a group to be efficient.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or Tensor(head_dim ** -0.5, mindspore.float32)

        self.q = nn.Dense(dim, dim, has_bias=qkv_bias)
        self.k = nn.Dense(dim, dim, has_bias=qkv_bias)
        self.v = nn.Dense(dim, dim, has_bias=qkv_bias)

        self.attn_drop = nn.Dropout( keep_prob= 1.0 - attn_drop)
        self.proj = nn.Dense(dim, dim, has_bias= True)
        self.proj_drop = nn.Dropout( keep_prob= 1.0 - proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size = sr_ratio, stride = sr_ratio, pad_mode = "valid", has_bias=True)
            self.norm = nn.LayerNorm((dim,))

    def construct(self, x, H, W):
        B, N, C = x.shape
        q = ops.Reshape()(self.q(x), (B, N, self.num_heads, C // self.num_heads))
        q = ops.Transpose()(q, (0, 2, 1, 3) ) # B, n_head,N, head_dim

        if self.sr_ratio > 1:
            x_ = ops.Reshape()(ops.Transpose()(x, (0, 2, 1)), (B, C, H, W))
            x_ =  ops.Transpose()(ops.Reshape()(self.sr(x_), (B, C, -1)), (0, 2, 1)) 
            x_ = self.norm(x_)
            k = ops.Transpose()(ops.Reshape()(self.k(x_), (B, -1, self.num_heads, C // self.num_heads)), (0, 2, 1, 3))
            v = ops.Transpose()(ops.Reshape()(self.v(x_), (B, -1, self.num_heads, C // self.num_heads)), (0, 2, 1, 3))

        else:
            k = ops.Transpose()(ops.Reshape()(self.k(x), (B, -1, self.num_heads, C // self.num_heads)), (0, 2, 1, 3))
            v = ops.Transpose()(ops.Reshape()(self.v(x), (B, -1, self.num_heads, C // self.num_heads)), (0, 2, 1, 3))

        attn = ops.Mul()(ops.BatchMatMul(transpose_b=True)(q, k), self.scale) 

        attn = nn.Softmax(axis=-1)(attn)
        attn = self.attn_drop(attn) 

        x = ops.Reshape()(ops.Transpose()(ops.BatchMatMul()(attn, v), (0, 2, 1, 3)), (B,N,C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GroupBlock(MindBlock):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, ws=1):
        super(GroupBlock, self).__init__(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop,
                                         drop_path, act_layer, norm_layer)
        del self.attn
        if ws == 1:
            self.attn = Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, sr_ratio)    # GSA
        else:
            self.attn = GroupAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, ws)     # LSA

    def construct(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Cell):
    """ 
    Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
                    f"img_size {img_size} should be divided by patch_size {patch_size}."

        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size = patch_size, stride= patch_size, pad_mode = "valid", has_bias=True)
        self.norm = nn.LayerNorm((embed_dim,))


    def construct(self, x):
        _, _, H, W = x.shape
        x = self.proj(x)

        b, c, h, w = x.shape
        x = ops.Transpose()(ops.Reshape()(x, (b, c, h * w)), (0, 2, 1))
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, (H, W)
    

# PEG  from https://arxiv.org/abs/2102.10882
class PosCNN(nn.Cell):
    def __init__(self, in_chans, embed_dim=768, s=1):
        super(PosCNN, self).__init__()
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size = 3, stride = s, \
                pad_mode = "pad", padding = 1, has_bias= True, group = embed_dim)
        self.s = s

    def construct(self, x, H, W):
        B, _, C = x.shape
        feat_token = x
        cnn_feat = ops.Reshape()(ops.Transpose()(feat_token, (0, 2, 1)), (B, C, H, W))

        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)

        b, c, h, w = x.shape
        x = ops.Reshape()(x, (b, c, h * w))
        x = ops.Transpose()(x, (0, 2, 1))
        return x


class ALTGVT(nn.Cell):
    """
    alias Twins-SVT
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256],
                 num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[4, 4, 4], sr_ratios=[4, 2, 1], block_cls = GroupBlock, wss=[7, 7, 7]):
        super(ALTGVT, self).__init__()
        
        self.depths = depths

        # patch_embed
        self.patch_embeds = nn.CellList()
        self.pos_drops = nn.CellList()

        for i in range(len(depths)):
            if i == 0:
                self.patch_embeds.append(PatchEmbed(img_size, patch_size, in_chans, embed_dims[i]))
            else:
                self.patch_embeds.append(
                    PatchEmbed(img_size // patch_size // 2 ** (i - 1), 2, embed_dims[i - 1], embed_dims[i]))
            self.pos_drops.append(nn.Dropout( keep_prob = 1.0 - drop_rate))

        # transformer encoder
        dpr = [x.item() for x in np.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.blocks = nn.CellList()
        for k in range(len(depths)):
            _block = nn.CellList([block_cls(
                dim=embed_dims[k], num_heads=num_heads[k], mlp_ratio=mlp_ratios[k], qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
                sr_ratio=sr_ratios[k], ws=1 if i % 2 == 1 else wss[k]) for i in range(depths[k])])
            self.blocks.append(_block)
            cur += depths[k]

        # PEG
        self.pos_block = nn.CellList(
            [PosCNN(embed_dim, embed_dim) for embed_dim in embed_dims]
        )

        self.norm = norm_layer((embed_dims[-1],))
        self.avgpool = P.ReduceMean(keep_dims=False)
        # classification head
        self.head = nn.Dense(in_channels = embed_dims[-1], out_channels = num_classes, has_bias= True) if num_classes > 0 else Identity()
        self.init_weights()

    def no_weight_decay(self):
        return ['%d.proj.weight' % i for i in range(4)]
        
    def init_weights(self):
        """
        init_weights

        """
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(initializer.initializer(initializer.TruncatedNormal(sigma=0.02),  # initializer.TruncatedNormal(sigma=0.02), TruncNormal(std=.02)
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
                if isinstance(cell, nn.Dense) and cell.bias is not None:
                    cell.bias.set_data(initializer.initializer(initializer.Zero(),
                                                               cell.bias.shape,
                                                               cell.bias.dtype))
            
            elif isinstance(cell, nn.Conv2d):
                fan_out = cell.kernel_size[0] * cell.kernel_size[1] * cell.out_channels
                fan_out //= cell.group
                cell.weight.set_data(initializer.initializer(initializer.Normal(sigma=math.sqrt(2.0 / fan_out)),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
                                                             
                if isinstance(cell, nn.Conv2d) and cell.bias is not None:
                    cell.bias.set_data(initializer.initializer(initializer.Zero(),
                                                               cell.bias.shape,
                                                               cell.bias.dtype))      

            elif isinstance(cell, nn.LayerNorm):
                cell.gamma.set_data(initializer.initializer(initializer.One(),
                                                            cell.gamma.shape,
                                                            cell.gamma.dtype))
                cell.beta.set_data(initializer.initializer(initializer.Zero(),
                                                           cell.beta.shape,
                                                           cell.beta.dtype))

    def forward_features(self, x):

        B = x.shape[0]
        for i in range(len(self.depths)):
            x, (H, W) = self.patch_embeds[i](x)
            x = self.pos_drops[i](x)
            for j, blk in enumerate(self.blocks[i]):
                x = blk(x, H, W)
                if j == 0:
                    x = self.pos_block[i](x, H, W)  # PEG here
            if i < len(self.depths) - 1:
                x = ops.Transpose()(ops.Reshape()(x, (B, H, W, -1)), (0, 3, 1, 2))

        x = self.norm(x)

        # GAP here
        x = self.avgpool(ops.Transpose()(x, (0, 2, 1)), 2)  # B C       
        return  x 
    
    def construct(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

def alt_gvt_small(pretrained=False, **kwargs):
    model = ALTGVT(
        patch_size=4, embed_dims=[64, 128, 256, 512], num_heads=[2, 4, 8, 16], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), depths=[2, 2, 10, 4], wss=[7, 7, 7, 7], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    return model


def alt_gvt_base(pretrained=False, **kwargs):
    model = ALTGVT(
        patch_size=4, embed_dims=[96, 192, 384, 768], num_heads=[3, 6, 12, 24], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), depths=[2, 2, 18, 2], wss=[7, 7, 7, 7], sr_ratios=[8, 4, 2, 1],
        **kwargs)

    return model


def alt_gvt_large(pretrained=False, **kwargs):
    model = ALTGVT(
        patch_size=4, embed_dims=[128, 256, 512, 1024], num_heads=[4, 8, 16, 32], mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        norm_layer= partial(nn.LayerNorm, epsilon=1e-6), depths=[2, 2, 18, 2], wss=[7, 7, 7, 7], sr_ratios=[8, 4, 2, 1],
        **kwargs)

    return model

if __name__ == "__main__":
    from mindspore import context
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    model = alt_gvt_small()
    skip_list = ()
    if hasattr(model, 'no_weight_decay'):
        skip_list = model.no_weight_decay()
        print("model_no_weight_decay: ",skip_list)

    for x in model.trainable_params():
        parameter_name = x.name

        if len(x.shape) == 1 or parameter_name.endswith(".bias") or parameter_name in skip_list:
             if parameter_name in skip_list:
                 print("para: ", parameter_name)

    x = Tensor(np.ones([32, 3, 224,224]), mindspore.float32)
    y = model(x)
    print("x.shape: ",y.type)