import torch
from torch import nn
from torchvision.models import resnet50

from bottleneck_transformer_pytorch import BottleStack

class BotNet(nn.Module):

    def __init__(self):
        super(BotNet, self).__init__()
        self.num_classes = 2
        self.botstack_layer = BottleStack(
            dim=256,
            fmap_size=24,  # set specifically for Pcam's 96 x 96
            dim_out=1024,
            proj_factor=4,
            downsample=True,
            heads=4,
            dim_head=128,
            rel_pos_emb=True,
            activation=nn.ReLU()
        )
        # model surgery
        self.backbone = list(resnet50().children())
        self.bot_net = nn.Sequential(
            *self.backbone[:5],
            self.botstack_layer,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1))

        self.fc_layer = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, self.num_classes),
            nn.Softmax(dim=1))

    def forward(self, x):
        bot_net_out = self.bot_net(x)
        out = self.fc_layer(bot_net_out)
        return out

# x = torch.randn(2, 3, 96, 96,device=device)
# layers = list(model.modules())
# for i, _ in enumerate(layers):
#     print(x.shape)
#     print(layers[0][i])
#     x = layers[0][i](x)
#     print(x.shape)

# use the 'BotNet'

# model = BotNet()
# model.to(device)
# preds = model(x)
# print(preds)
