from tsai.all import *
import torch
import numpy as np
"""
c_in -  No. of Channels
c_out - No. of target classes - 2 for binary classification
seq_len - 10*frequency
d_model - 128
depth - 6
n_heads - 8
mlp_ratio - 3
token_size - 50
tokenizer - 

TSiTPlus(c_in:12, c_out:1, seq_len:int, d_model:int=128, depth:int=6,
           n_heads:int=16, act:str='gelu', lsa:bool=False,
           attn_dropout:float=0.0, dropout:float=0.0,
           drop_path_rate:float=0.0, mlp_ratio:int=1, qkv_bias:bool=True,
           pre_norm:bool=False, use_token:bool=False, use_pe:bool=True,
           cat_pos:Optional[list]=None, n_cat_embeds:Optional[list]=None,
           cat_embed_dims:Optional[list]=None,
           cat_padding_idxs:Optional[list]=None, token_size:int=None,
           tokenizer:Optional[Callable]=None,
           feature_extractor:Optional[Callable]=None, flatten:bool=False,
           concat_pool:bool=True, fc_dropout:float=0.0, use_bn:bool=False,
           bias_init:Union[float,list,NoneType]=None,
           y_range:Optional[tuple]=None,
           custom_head:Optional[Callable]=None, verbose:bool=True,
           **kwargs)

"""
c_in = 12
c_out = 2
seq_len = 1000
bs = 16
"""
Model parameters :

MAE - Pretrained ViT size
d_model = 128
depth = 6
n_heads = 8
mlp_ratio = 3
token_size = 50
No. of Model Parameters - 1.08M

VIT Base parameters
d_model = 768
depth = 12
n_heads = 12
mlp_ratio = 4
token_size = 50
No. of Model Parameters - 86M

ViT Large Parameters 
d_model = 1024
depth = 24
n_heads = 16
mlp_ratio = 4
token_size = 50
No. of Model Parameters - 307M

ViT Huge Parameters
d_model = 1280
depth = 32
n_heads = 16
mlp_ratio = 4
token_size = 50
No. of Model Parameters - 630M
"""

d_model = 1280
depth = 32
n_heads = 16
mlp_ratio = 4
token_size = 50

xb = torch.rand(bs, c_in, seq_len)
bias_init = np.array([0.8, .2])
model = TSiTPlus(c_in = c_in, c_out=c_out, seq_len=seq_len,d_model=d_model, depth = depth, n_heads = n_heads, mlp_ratio = mlp_ratio, token_size=token_size)

# test_eq(model.head[1].bias.data, tensor(bias_init))
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

print(count_parameters(model))