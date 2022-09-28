from torch import nn
import torchvision as tv
from custom_network.Network import Network


class CustomModels:

    def __init__(self, IN_CHANNEL, NUM_OUTPUT):
        self.input_channel = IN_CHANNEL
        self.n_class_output = NUM_OUTPUT

        self.cifar10_model = nn.Sequential(
            nn.Conv2d(self.input_channel, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 64 x 16 x 16
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 128 x 8 x 8
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 256 x 4 x 4
            nn.BatchNorm2d(256),

            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.n_class_output)
        )

        self.model_25k_wo_dw = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channel, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),

            nn.Dropout(0.05),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Dropout(0.17),
            nn.MaxPool2d(2, 2),

            nn.AvgPool2d(kernel_size=4),
            nn.Flatten(),
            nn.Linear(64, self.n_class_output)
        )

        self.model_25k_w_dw = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channel, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            # 32 -> 30
            nn.ReLU(),
            nn.BatchNorm2d(8),
            # commented by sabeesh - intial layers with depth wise convs
            # nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=0,groups = 8, bias=False), # 30 -> 28
            # nn.Conv2d(in_channels=8, out_channels=14, kernel_size=(1, 1), padding=0, bias=False),

            # new layers- bharath
            nn.Conv2d(in_channels=8, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14),
            nn.Dropout(0.05),
            nn.MaxPool2d(2, 2),  # 28 -> 14

            nn.Conv2d(in_channels=14, out_channels=14, kernel_size=(3, 3), padding=0, groups=14, bias=False),
            # 14 -> 12
            nn.Conv2d(in_channels=14, out_channels=24, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(24),

            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3, 3), padding=0, groups=24, bias=False),
            # 12 -> 10
            nn.Conv2d(in_channels=24, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            # nn.MaxPool2d(2, 2)

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, groups=32, bias=False),  # 10 -> 8
            nn.Conv2d(in_channels=32, out_channels=42, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(42),

            nn.Conv2d(in_channels=42, out_channels=54, kernel_size=(3, 3), padding=0, bias=False),  # 8 -> 4
            nn.ReLU(),
            nn.BatchNorm2d(54),
            nn.Dropout(0.08),

            # nn.AvgPool2d(kernel_size=4),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(54, self.n_class_output)
        )

        self.model_143k_wo_dw = nn.Sequential(
            nn.Conv2d(self.input_channel, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),

            nn.Dropout2d(0.06),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.07),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.07),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 120, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(120),

            nn.AvgPool2d(2),
            nn.Flatten(),
            nn.Linear(120, self.n_class_output)
        )

        self.model_143k_w_dw = nn.Sequential(
            nn.Conv2d(self.input_channel, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),

            nn.Dropout2d(0.06),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, groups=16),

            nn.Conv2d(16, 32, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.07),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32),
            nn.Conv2d(32, 64, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.07),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64),
            nn.Conv2d(64, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128),
            nn.Conv2d(128, 192, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(192),
            nn.Dropout2d(0.05),

            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1, groups=192),
            nn.Conv2d(192, 260, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(260),

            nn.AvgPool2d(3),
            nn.Flatten(),
            nn.Linear(260, self.n_class_output)
        )

        self.model_340k_wo_dw = nn.Sequential(

            nn.Conv2d(self.input_channel, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.06),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.07),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(96),

            nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.Dropout2d(0.07),
            nn.MaxPool2d(2),

            nn.Conv2d(96, 160, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(160),

            nn.AvgPool2d(2),
            nn.Flatten(),
            nn.Linear(160, self.n_class_output)
        )

        self.model_340k_w_dw = nn.Sequential(

            nn.Conv2d(self.input_channel, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.06),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32),
            nn.Conv2d(32, 64, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.07),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64),
            nn.Conv2d(64, 96, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(96),

            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.Dropout2d(0.07),
            nn.MaxPool2d(2),

            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, groups=96),
            nn.Conv2d(96, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128),
            nn.Conv2d(128, 192, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(192),
            nn.Dropout2d(0.05),

            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1, groups=192),
            nn.Conv2d(192, 210, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(210),

            nn.Conv2d(210, 210, kernel_size=3, stride=1, padding=1, groups=210),
            nn.Conv2d(210, 240, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(240),
            nn.Dropout2d(0.02),

            nn.Conv2d(240, 240, kernel_size=3, stride=1, padding=1, groups=240),
            nn.Conv2d(240, 280, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(280),

            # nn.AvgPool2d(3),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(280, self.n_class_output)
        )

        self.model_600k_wo_dw = nn.Sequential(

            nn.Conv2d(self.input_channel, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.06),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.07),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.07),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.AvgPool2d(2),
            nn.Flatten(),
            nn.Linear(256, self.n_class_output)
        )

        self.model_600k_w_dw = nn.Sequential(

            nn.Conv2d(self.input_channel, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.03),
            # nn.MaxPool2d(2),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32),
            nn.Conv2d(32, 64, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.03),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64),
            nn.Conv2d(64, 96, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(96),

            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.Dropout2d(0.07),

            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, groups=96),
            nn.Conv2d(96, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128),
            nn.Conv2d(128, 192, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(192),
            nn.Dropout2d(0.05),
            nn.MaxPool2d(2),

            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1, groups=192),
            nn.Conv2d(192, 260, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(260),

            nn.Conv2d(260, 260, kernel_size=3, stride=1, padding=1, groups=260),
            nn.Conv2d(260, 320, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(320),
            nn.Dropout2d(0.02),
            nn.MaxPool2d(2),

            nn.Conv2d(320, 320, kernel_size=3, stride=1, padding=1, groups=320),
            nn.Conv2d(320, 370, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(370),

            nn.Conv2d(370, 370, kernel_size=3, stride=1, padding=1, groups=370),
            nn.Conv2d(370, 415, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(415),

            # nn.AvgPool2d(3),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(415, self.n_class_output)
        )

        self.model_1M_wo_dw = nn.Sequential(
            ################################## 1 ST CONVOLUTIONAL BLOCK #####################################
            nn.Conv2d(in_channels=self.input_channel, dilation=1, out_channels=16, padding=1, kernel_size=3),
            # in 32, out 32, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=32, dilation=1, padding=1, kernel_size=(3, 3)),
            # in 32, out 32, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, dilation=1, padding=1, kernel_size=(3, 3)),
            # in 32, out 32, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.06),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=32, out_channels=64, dilation=1, padding=1, kernel_size=(3, 3)),
            # in 16, out 16, RF ?
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, dilation=1, padding=1, kernel_size=(3, 3)),
            # in 16, out 16, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.07),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=64, out_channels=128, dilation=1, padding=1, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, dilation=2, padding=2, kernel_size=(3, 3)),
            # in 8, out 8, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.07),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=128, out_channels=256, dilation=1, padding=1, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256, dilation=1, padding=1, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.AvgPool2d(kernel_size=3),  # 1x1/15
            nn.Flatten(),
            nn.Linear(256, self.n_class_output)
        )

        self.model_1M_w_dw = nn.Sequential(
            ################################## 1 ST CONVOLUTIONAL BLOCK #####################################
            nn.Conv2d(in_channels=self.input_channel, out_channels=16, dilation=1, padding=1, kernel_size=(3, 3)),
            # in 32, out 32, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(16),  # 26x26 /8
            nn.Conv2d(in_channels=16, out_channels=32, dilation=1, padding=1, kernel_size=(3, 3)),
            # in 32, out 32, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.03),
            # nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=32, out_channels=32, groups=32, dilation=1, padding=1, kernel_size=(3, 3)),
            nn.Conv2d(in_channels=32, out_channels=64, dilation=1, padding=0, kernel_size=(1, 1)),  # 8, 8, 3
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, dilation=1, padding=1, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.03),
            nn.MaxPool2d(2, 2),  # in 32, out 16, RF 3

            nn.Conv2d(in_channels=64, out_channels=64, groups=64, dilation=1, padding=1, kernel_size=(3, 3)),
            nn.Conv2d(in_channels=64, out_channels=96, dilation=1, padding=0, kernel_size=(1, 1)),  # 8, 8, 3
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.Conv2d(in_channels=96, out_channels=96, dilation=2, padding=2, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.Dropout(0.07),
            # nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=96, out_channels=96, groups=96, dilation=1, padding=1, kernel_size=(3, 3)),
            nn.Conv2d(in_channels=96, out_channels=160, dilation=1, padding=0, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(160),
            nn.Conv2d(in_channels=160, out_channels=160, groups=160, dilation=1, padding=1, kernel_size=(3, 3)),
            nn.Conv2d(in_channels=160, out_channels=220, dilation=1, padding=0, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(220),
            nn.Dropout(0.05),
            nn.MaxPool2d(2, 2),  # in 16, out 8, RF 3

            nn.Conv2d(in_channels=220, out_channels=220, groups=220, dilation=1, padding=1, kernel_size=(3, 3)),
            nn.Conv2d(in_channels=220, out_channels=300, dilation=1, padding=0, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(300),
            nn.Conv2d(in_channels=300, out_channels=300, groups=300, dilation=1, padding=1, kernel_size=(3, 3)),
            nn.Conv2d(in_channels=300, out_channels=380, dilation=1, padding=0, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(380),
            nn.Dropout(0.02),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(in_channels=380, out_channels=380, groups=380, dilation=1, padding=1, kernel_size=(3, 3)),
            nn.Conv2d(in_channels=380, out_channels=460, dilation=1, padding=0, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(460),
            # nn.Dropout(0.02),
            nn.Conv2d(in_channels=460, out_channels=460, groups=460, dilation=1, padding=1, kernel_size=(3, 3)),
            nn.Conv2d(in_channels=460, out_channels=540, dilation=1, padding=0, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(540),
            nn.Conv2d(in_channels=540, out_channels=540, groups=540, dilation=1, padding=1, kernel_size=(3, 3)),
            nn.Conv2d(in_channels=540, out_channels=620, dilation=1, padding=0, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(620),

            # nn.Conv2d(in_channels = 460,out_channels = 460, groups = 460, dilation  = 1,padding = 1,kernel_size= (3,3)),    # in 4, out 4,, RF ?
            # nn.Conv2d(in_channels = 460,out_channels = 520, dilation  = 1,padding = 0,kernel_size= (1,1)),                  # in 4, out 4, RF ?
            # nn.ReLU(),
            # nn.BatchNorm2d(520),
            nn.AdaptiveAvgPool2d(1),
            # nn.AvgPool2d(kernel_size=27),  # 1x1/15
            nn.Flatten(),
            nn.Linear(620, self.n_class_output)
        )

    def init_model(self, model_name):
        if model_name == 'resnet_18':
            model = tv.models.resnet18()

        elif model_name == 'resnet_34':
            model = tv.models.resnet34()

        elif model_name == 'resnet_50':
            model = tv.models.resnet50()

        elif model_name == 'resnet_101':
            model = tv.models.resnet101()

        elif model_name == 'resnet_152':
            model = tv.models.resnet152()

        elif model_name == 'resnext101_32x8d':
            model = tv.models.resnext101_32x8d()

        elif model_name == 'wide_resnet101_2':
            model = tv.models.wide_resnet101_2()

        elif model_name == 'wide_resnet50_2':
            model = tv.models.wide_resnet50_2()

        elif model_name == 'model_25k_w_dw':
            model = Network(self.model_25k_w_dw)

        elif model_name == 'model_25k_wo_dw':
            model = Network(self.model_25k_wo_dw)

        elif model_name == 'model_143k_w_dw':
            model = Network(self.model_143k_w_dw)

        elif model_name == 'model_143k_wo_dw':
            model = Network(self.model_143k_wo_dw)

        elif model_name == 'model_340k_w_dw':
            model = Network(self.model_340k_w_dw)

        elif model_name == 'model_340k_wo_dw':
            model = Network(self.model_340k_wo_dw)

        elif model_name == 'model_600k_w_dw':
            model = Network(self.model_600k_w_dw)

        elif model_name == 'model_600k_wo_dw':
            model = Network(self.model_600k_wo_dw)

        elif model_name == 'model_1M_w_dw':
            model = Network(self.model_1M_w_dw)

        elif model_name == 'model_1M_wo_dw':
            model = Network(self.model_1M_wo_dw)

        else:
            raise ValueError("No Model to Process")

        if 'resnet' in model_name:
            model.conv1 = nn.Conv2d(self.input_channel,
                                    model.conv1.out_channels,
                                    kernel_size=model.conv1.kernel_size,
                                    stride=model.conv1.stride,
                                    padding=model.conv1.padding,
                                    bias=model.conv1.bias)
            model.fc = nn.Linear(in_features=model.fc.in_features, out_features=self.n_class_output, bias=True)

        model.__setattr__('_name', model_name)

        return model


