import torch
from torch import nn
import torch.nn.functional as F
# from torchvision.models import resnet50,densenet121
# from botnet_resnet.ensemble_main import EnsembleBotResNet
# from bottleneck_transformer_pytorch.botnet_main import BotNet
# from base_vit.vit import ViT
# from custom_network.CustomCnnNetwork import CustomCNN

class MyEnsemble(nn.Module):
    def __init__(self, modelA, modelB,modelC,modelD):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC
        self.modelD = modelD
        self.classifier = nn.Linear(8, 2)

    def forward(self, x):
        x1 = self.modelA(x)
        x2 = self.modelB(x)
        x3 = self.modelC(x)
        x4 = self.modelD(x)
        x = torch.cat((x1,x2,x3,x4), dim=1)
        x = self.classifier(F.relu(x))
        return x


# Create models and load state_dicts
# modelA = densenet121()
# modelB = BotNet()
# modelC = CustomCNN()
# modelD = EnsembleBotResNet()
# # Load state dicts
# modelA.load_state_dict(torch.load("/Users/eshwarmurthy/Desktop/personal/Msc-LJMU/Pcam_data/model_output/Densnet/checkpoint_best_epoch_num_6_acc_82.59.pth"))
# modelB.load_state_dict(torch.load("/Users/eshwarmurthy/Desktop/personal/Msc-LJMU/Pcam_data/model_output/Botnet_old/checkpoint_best_epoch_num_1_acc_80.5.pth"))
# modelC.load_state_dict(torch.load("/Users/eshwarmurthy/Desktop/personal/Msc-LJMU/Pcam_data/model_output/CustomCNN/checkpoint_best_epoch_num_3_acc_84.06.pth"))
# modelD.load_state_dict(torch.load("/Users/eshwarmurthy/Desktop/personal/Msc-LJMU/Pcam_data/model_output/EnsembleBotResnet/checkpoint_best_epoch_num_1_acc_81.11.pth"))
# model = MyEnsemble(modelA, modelB,modelC,modelD)
# device=torch.device("mps")
# model.to(device)
# x = torch.randn(2, 3, 156, 156,device=device)
# output = model(x)