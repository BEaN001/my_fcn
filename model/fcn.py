import torch
import torch.nn as nn
import sys
sys.path.append("..")
from src.utils import bilinear_kernel


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


class Layer(nn.Module):
    def __init__(self, in_channels, layer_list):
        super(Layer, self).__init__()
        self.layer = self.make_layers(in_channels, layer_list)

    def make_layers(self, in_channels, layer_list):
        layers = []
        for out_channels in layer_list:
            layers += [Block(in_channels, out_channels)]
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer(x)
        return out


# VGG
class VGG(nn.Module):
    """
    VGG model
    """
    def __init__(self, n_class=21):
        super(VGG, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.layer1 = Layer(64, [64])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2 = Layer(64, [128, 128])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer3 = Layer(128, [256, 256, 256, 256])
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer4 = Layer(256, [512, 512, 512, 512])
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer5 = Layer(512, [512, 512, 512, 512])
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # modify to be compatible with segmentation and classification
        self.fc6 = nn.Linear(512*7*7, 4096)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout()

        self.fc7 = nn.Linear(4096, 4096)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout()

        self.score = nn.Linear(4096, n_class)

    def forward(self, x):
        f0 = self.relu1(self.bn1(self.conv1(x)))
        f1 = self.pool1(self.layer1(f0))
        f2 = self.pool2(self.layer2(f1))
        f3 = self.pool3(self.layer3(f2))
        f4 = self.pool4(self.layer4(f3))
        f5 = self.pool5(self.layer5(f4))
        f5 = f5.view(f5.size(0), -1)
        f6 = self.drop6(self.relu6(self.fc6(f5)))
        f7 = self.drop7(self.relu7(self.fc7(f6)))
        score = self.score(f7)
        return score


class VGG_fcn32s(nn.Module):
    """
    VGG_fcn_32s
    """

    def __init__(self, n_class=21):
        super(VGG_fcn32s, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=100)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.layer1 = Layer(64, [64])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2 = Layer(64, [128, 128])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer3 = Layer(128, [256, 256, 256, 256])
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer4 = Layer(256, [512, 512, 512, 512])
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer5 = Layer(512, [512, 512, 512, 512])
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # modify to be compatible with segmentation and classification
        # self.fc6 = nn.Linear(512*7*7, 4096)
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout()

        # self.fc7 = nn.Linear(4096, 4096)
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout()

        # self.score = nn.Linear(4096, n_class)
        self.score = nn.Conv2d(4096, n_class, 1)

        self.upscore = nn.ConvTranspose2d(n_class, n_class, 64, 32)

    def forward(self, x):
        f0 = self.relu1(self.bn1(self.conv1(x)))
        # print(f'f0.shape: {f0.shape}')
        f1 = self.pool1(self.layer1(f0))
        # print(f'f1.shape: {f1.shape}')
        f2 = self.pool2(self.layer2(f1))
        # print(f'f2.shape: {f2.shape}')
        f3 = self.pool3(self.layer3(f2))
        # print(f'f3.shape: {f3.shape}')
        f4 = self.pool4(self.layer4(f3))
        # print(f'f4.shape: {f4.shape}')
        f5 = self.pool5(self.layer5(f4))
        # f5 = f5.view(f5.size(0), -1)
        # print(f'f5.shape: {f5.shape}')
        f6 = self.drop6(self.relu6(self.fc6(f5)))
        # print(f'f6.shape: {f6.shape}')
        f7 = self.drop7(self.relu7(self.fc7(f6)))
        # print(f'f7.shape: {f7.shape}')
        score = self.score(f7)  # 21*7*7
        # print(f'score.shape: {score.shape}')
        upscore = self.upscore(score)  # 21*256*256
        # print(f'upscore.shape: {upscore.shape}')
        upscore = upscore[:, :, 19:19 + x.size(2), 19:19 + x.size(3)].contiguous()
        # print(f'upscore.shape: {upscore.shape}')
        return upscore


class VGG_19bn_8s(nn.Module):
    def __init__(self, n_class=21):
        super(VGG_19bn_8s, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=100)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.layer1 = Layer(64, [64])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2 = Layer(64, [128, 128])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer3 = Layer(128, [256, 256, 256, 256])
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer4 = Layer(256, [512, 512, 512, 512])
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer5 = Layer(512, [512, 512, 512, 512])
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc6 = nn.Conv2d(512, 4096, 7)  # padding=0
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, n_class, 1)
        self.trans_f4 = nn.Conv2d(512, n_class, 1)
        self.trans_f3 = nn.Conv2d(256, n_class, 1)

        self.up2times = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)
        self.up4times = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)
        self.up32times = nn.ConvTranspose2d(
            n_class, n_class, 16, stride=8, bias=False)
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data = bilinear_kernel(n_class, n_class, m.kernel_size[0])

    def forward(self, x):
        f0 = self.relu1(self.bn1(self.conv1(x)))  # s: 64*422*422
        # print(f'f0.shape: {f0.shape}')
        f1 = self.pool1(self.layer1(f0))  # s/2:64*211*211
        # print(f'f1.shape: {f1.shape}')
        f2 = self.pool2(self.layer2(f1))  # s/4: 128*105*105
        # print(f'f2.shape: {f2.shape}')
        f3 = self.pool3(self.layer3(f2))  # s/8:256*52*52
        # print(f'f3.shape: {f3.shape}')
        f4 = self.pool4(self.layer4(f3))  # s/16: 512*26*26
        # print(f'f4.shape: {f4.shape}')
        f5 = self.pool5(self.layer5(f4))  # s/32: 512*13*13
        # print(f'f5.shape: {f5.shape}')

        f6 = self.drop6(self.relu6(self.fc6(f5)))  # 4096*7*7
        # print(f'f6.shape: {f6.shape}')
        f7 = self.score_fr(self.drop7(self.relu7(self.fc7(f6))))  # 21*7*7
        # print(f'f7.shape: {f7.shape}')

        up2_feat = self.up2times(f7)  # 21*16*16   2*conv7
        # print(f'up2_feat.shape: {up2_feat.shape}')
        h = self.trans_f4(f4)  # 21*26*26 pool4
        # print(f'h.shape: {h.shape}')
        h = h[:, :, 5:5 + up2_feat.size(2), 5:5 + up2_feat.size(3)]  # 21*16*16 croped pool4
        # print(f'h.shape: {h.shape}')
        h = h + up2_feat  # 21*16*16 croped poo4 + 2*conv7
        # print(f'h.shape: {h.shape}')

        up4_feat = self.up4times(h)  # 21*34*34  2*(croped poo4 + 2*conv7)
        # print(f'up4_feat.shape: {up4_feat.shape}')
        h = self.trans_f3(f3)  # 21*52*52 pool3
        # print(f'h.shape: {h.shape}')
        h = h[:, :, 9:9 + up4_feat.size(2), 9:9 + up4_feat.size(3)]  # 21*34*34 croped pool3
        # print(f'h.shape: {h.shape}')
        h = h + up4_feat  # 21*34*34 croped pool3 + 2*(croped poo4 + 2*conv7)
        # print(f'h.shape: {h.shape}')

        h = self.up32times(h)  # 21*280*280 score
        # print(f'h.shape: {h.shape}')
        final_scores = h[:, :, 31:31 + x.size(2), 31:31 + x.size(3)].contiguous()  # 21*280*280 croped score
        # print(f'final_scores.shape: {final_scores.shape}')

        return final_scores


if __name__ == "__main__":
    from torchsummary import summary
    print('*'*30)
    print('VGG!')
    vgg_model = VGG()
    x = torch.randn((2, 3, 224, 224), dtype=torch.float32)
    y = vgg_model(x)
    print(y.shape)
    # summary_vision = summary(vgg_model, (3, 224, 224))

    print('*' * 30)
    print('VGG_19bn_fcn_32s!')
    vgg_fcn32s_model = VGG_fcn32s()
    x = torch.randn((2, 3, 224, 224), dtype=torch.float32)
    y = vgg_fcn32s_model(x)
    print(y.shape)
    # summary_vision = summary(vgg_fcn32s_model, (3, 224, 224))

    print('*' * 30)
    print('VGG_19bn_fcn_8s!')
    model = VGG_19bn_8s(21)
    x = torch.randn(2, 3, 224, 224, dtype=torch.float32)
    #model.eval()
    y_vgg = model(x)
    print(y_vgg.shape)

