import torch
from torch.nn import functional as F
from torch import Tensor
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models


class RL1Loss(nn.Module):
    """
    偏移损失计算
    """
    def __init__(self, roll_dis: int):
        super(RL1Loss, self).__init__()
        self.roll_dis = roll_dis  # 偏移距离设定

    def initialize(self, loss):
        with torch.no_grad():
            self.criterion = loss  # 偏移损失中，用于计算每次偏移后的损失，loss输入为L1Loss

    def forward(self, _input: Tensor, target: Tensor) -> Tensor:
        """
        在偏移距离范围内，使用torch.roll实现偏移操作，然后计算偏移后的损失，返回多次偏移的最小损失
        """
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
            self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

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
