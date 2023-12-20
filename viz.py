from graphviz import Digraph
import torch
from torch.autograd import Variable

# make_dot was moved to https://github.com/szagoruyko/pytorchviz
from torchviz import make_dot
from models_mae import mae_vit_1dcnn

x = torch.rand(10, 1, 12, 1000)
model = mae_vit_1dcnn()
w = model(x)
print(w)