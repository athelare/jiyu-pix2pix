import torch
from torch.nn import functional as F
from torch import Tensor
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models


class RL1Loss(nn.Module):
    def __init__(self, roll_dis: int):
        super(RL1Loss, self).__init__()
        self.roll_dis = roll_dis

    def initialize(self, loss):
        with torch.no_grad():
            self.criterion = loss

    def forward(self, _input: Tensor, target: Tensor) -> Tensor:
        losses_list = []
        for i in range(-self.roll_dis,self.roll_dis+1):
            for j in range(-self.roll_dis,self.roll_dis+1):
                tmp_in = torch.roll(_input, (i, j), (0, 1))
                losses_list.append(self.criterion(tmp_in, target, reduction='mean'))
        return torch.min(torch.stack(losses_list))


class PerceptualLoss:

    def contentFunc(self):
        conv_3_3_layer = 14
        cnn = models.vgg19(pretrained=True).features
        cnn = cnn.cuda()
        model = nn.Sequential()
        model = model.cuda()
        model = model.eval()
        for i, layer in enumerate(list(cnn)):
            model.add_module(str(i), layer)
            if i == conv_3_3_layer:
                break
        return model

    def initialize(self, loss):
        with torch.no_grad():
            self.criterion = loss
            self.contentFunc = self.contentFunc()
            # self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.transform = transforms.Normalize(mean=[0.5], std=[0.225])

    def get_loss(self, fakeIm, realIm):
        fakeIm = (fakeIm + 1) / 2.0
        realIm = (realIm + 1) / 2.0
        fakeIm[0, :, :, :] = self.transform(fakeIm[0, :, :, :])
        realIm[0, :, :, :] = self.transform(realIm[0, :, :, :])
        f_fake = self.contentFunc.forward(fakeIm)
        f_real = self.contentFunc.forward(realIm)
        f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake, f_real_no_grad)
        return 0.006 * torch.mean(loss) + 0.5 * nn.MSELoss()(fakeIm, realIm)

    def __call__(self, fakeIm, realIm):
        return self.get_loss(fakeIm, realIm)
