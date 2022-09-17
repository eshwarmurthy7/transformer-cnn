import torch
from torch import nn
from torchvision.models import resnet50

from bottleneck_transformer_pytorch.bottleneck_transformer_pytorch import BottleStack
num_classes = 2
layer = BottleStack(
    dim = 256,
    fmap_size = 24,        # set specifically for imagenet's 224 x 224
    dim_out = 1024,
    proj_factor = 4,
    downsample = True,
    heads = 4,
    dim_head = 128,
    rel_pos_emb = True,
    activation = nn.ReLU()
)

resnet = resnet50()

# model surgery

backbone = list(resnet.children())

model = nn.Sequential(
    *backbone[:5],
    layer,
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(1),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, num_classes),
    nn.Softmax(dim=1)
)

x = torch.randn(2, 3, 96, 96)
# layers = list(model.modules())
# for i, _ in enumerate(layers):
#     print(x.shape)
#     print(layers[0][i])
#     x = layers[0][i](x)
#     print(x.shape)

# use the 'BotNet'


preds = model(x) # (2, 1000)
print(preds)