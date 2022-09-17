import torch
from torch import nn
from torchvision.models import resnet50
from bottleneck_transformer_pytorch.bottleneck_transformer_pytorch import BottleStack

num_classes = 2

x = torch.randn(2, 3, 96, 96)


# layers = list(model.modules())
# for i, _ in enumerate(layers):
#     print(x.shape)
#     print(layers[0][i])
#     x = layers[0][i](x)
#     print(x.shape)

# use the 'BotNet'


class Resnet_50(nn.Module):
    def __init__(self):
        super(Resnet_50, self).__init__()
        self.norm = nn.BatchNorm2d(3)
        self.mobilenet = resnet50()
        self.mobilenet.classifier = nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        x = self.mobilenet(x)
        feature = x.reshape(-1, 1000)
        return feature


class Resnet_50_BotNet(nn.Module):
    def __init__(self):
        super(Resnet_50_BotNet, self).__init__()
        self.resnet_50 = Resnet_50()
        self.bot_net = nn.Sequential(
            *list(resnet50().children())[:5],
            BottleStack(
                dim=256,
                fmap_size=24,  # set specifically for imagenet's 224 x 224
                dim_out=1024,
                proj_factor=4,
                downsample=True,
                heads=4,
                dim_head=128,
                rel_pos_emb=True,
                activation=nn.ReLU()
            ),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
        )
        self.fc_layer = nn.Sequential(nn.Linear(2024, 512),
                                      nn.ReLU(),
                                      nn.Dropout(0.5),
                                      nn.Linear(512, num_classes),
                                      )

    def forward(self, x):
        resnet_50_out = self.resnet_50(x)
        bot_net_out = self.bot_net(x)
        feature = torch.cat((resnet_50_out, bot_net_out), axis=1)
        out = self.fc_layer(feature)
        return out


com_mod = Resnet_50_BotNet()
combined = com_mod(x)
print(combined.shape)
