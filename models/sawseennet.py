from torch import nn
from torch.nn import functional as F
import torch
from itertools import chain
import pretrainedmodels
from oc_module import asp_oc_block


MODEL_NAME = 'se_resnext50_32x4d'


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_, out, stride):
        super().__init__()
        self.conv = conv3x3(in_, out, stride)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilations=(1, 3, 5), batch_norm=False):
        super().__init__()
        self.conv_0 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2,
                                         padding=1, output_padding=1)

        self.batch_norm = batch_norm

        self.blocks = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=dilation, padding=dilation),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=dilation, padding=dilation),
                nn.ReLU(inplace=True)
            ) for dilation in dilations]
        )

    def forward(self, x):
        conv0 = self.conv_0(x)
        conv0 = F.relu(conv0, inplace=True)
        blocks = [self.blocks[index](conv0) for index in range(len(self.blocks))]
        return torch.cat(blocks, dim=1)


class SawSeenNet(nn.Module):
    def __init__(self, base_channels, pretrained=False, frozen=True):
        super(SawSeenNet, self).__init__()

        self.base_channels = base_channels
        self.pretrained = pretrained
        self.frozen = frozen
        self.training = False

        self.pool = nn.MaxPool2d(2, 2)

        self.probability = 0.2
        self.probability_class = 0.4

        self.dropout = F.dropout2d

        self.encoder = pretrainedmodels.__dict__[MODEL_NAME](num_classes=1000, pretrained='imagenet')

        if self.frozen:
            for p in self.encoder.parameters():
                p.data.requires_grad_(requires_grad=False)

        self.init_conv = self.encoder.layer0.conv1
        self.bn1 = self.encoder.layer0.bn1
        self.relu = self.encoder.layer0.relu1
        self.maxpool = self.encoder.layer0.pool

        self.enc_0 = self.encoder.layer1

        self.enc_1 = self.encoder.layer2

        self.enc_2 = self.encoder.layer3

        self.enc_3 = self.encoder.layer4

        self.middle_conv = ConvRelu(self.base_channels * 32, self.base_channels * 8, stride=2)
        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.class_conv_compressor = nn.Conv2d(512, 64, kernel_size=1)
        self.class_conv = nn.Conv2d(64, 1, kernel_size=1)

        self.dec_3 = DecoderBlock(self.base_channels * 8, self.base_channels * 4, dilations=(1,))
        self.dec_3_pred = nn.Conv2d(self.base_channels * 4, 1, kernel_size=3, padding=1)

        self.dec_2 = DecoderBlock(2304, self.base_channels * 2, dilations=(3,))
        self.dec_2_pred = nn.Conv2d(self.base_channels * 2, 1, kernel_size=3, padding=1)

        self.dec_1 = DecoderBlock(1152, self.base_channels * 1, dilations=(3,))
        self.dec_1_pred = nn.Conv2d(self.base_channels * 1, 1, kernel_size=3, padding=1)

        self.dec_0 = DecoderBlock(960, self.base_channels // 2, dilations=(5,))
        self.dec_0_pred = nn.Conv2d(self.base_channels // 2, 1, kernel_size=3, padding=1)

        self.dec_final_0 = DecoderBlock(480, self.base_channels // 2, dilations=(5,))

        self.context = asp_oc_block.ASP_OC_Module(128, 48)

        self.final = nn.Conv2d(48, 1, kernel_size=5, padding=2)

        self._init_weights()

    def forward(self, x):
        init_conv = self.init_conv(x)
        init_conv = self.bn1(init_conv)
        init_conv = self.relu(init_conv)

        enc_0 = self.enc_0(init_conv)

        enc_1 = self.enc_1(enc_0)

        enc_2 = self.enc_2(enc_1)

        enc_3 = self.enc_3(enc_2)

        middle_conv = self.middle_conv(enc_3)

        middle_pooling = self.avg_pooling(middle_conv)
        middle_pooling = F.relu(self.class_conv_compressor(middle_pooling), inplace=True)
        class_empty_pred = self.class_conv(
            self.dropout(middle_pooling,
                         p=self.probability_class, training=self.training)
        ).view(-1, 1)

        dec_3 = self.dec_3(middle_conv)
        dec_3_pred = self.dec_3_pred(
            self.dropout(dec_3, p=self.probability, training=self.training)
        )

        dec_3_cat = torch.cat([
            dec_3,
            enc_3
        ], 1)

        dec_2 = self.dec_2(dec_3_cat)
        dec_2_pred = self.dec_2_pred(
            self.dropout(dec_2, p=self.probability, training=self.training)
        )

        dec_2_cat = torch.cat([
            dec_2,
            enc_2
        ], 1)

        dec_1 = self.dec_1(dec_2_cat)
        dec_1_pred = self.dec_1_pred(
            self.dropout(dec_1, p=self.probability, training=self.training)
        )

        dec_1_cat = torch.cat([
            dec_1,
            F.interpolate(dec_2, scale_factor=2, mode='nearest'),
            F.interpolate(dec_3, scale_factor=4, mode='nearest'),
            enc_1
        ], 1)

        dec_0 = self.dec_0(dec_1_cat)
        dec_0_pred = self.dec_0_pred(
            self.dropout(dec_0, p=self.probability, training=self.training)
        )

        dec_0_cat = torch.cat([
            dec_0,
            F.interpolate(dec_1, scale_factor=2, mode='nearest'),
            F.interpolate(dec_2, scale_factor=4, mode='nearest'),
            enc_0
        ], 1)

        dec_final_0 = self.dec_final_0(dec_0_cat)

        hyper_column = torch.cat([
            self.dropout(dec_final_0,
                         p=self.probability, training=self.training),

            self.dropout(F.interpolate(dec_0, scale_factor=2, mode='nearest'),
                         p=self.probability, training=self.training),

            self.dropout(F.interpolate(dec_1, scale_factor=4, mode='nearest'),
                         p=self.probability, training=self.training),
        ], dim=1)

        oc = self.context(hyper_column)
        final = self.final(oc)

        return final, class_empty_pred, dec_0_pred, dec_1_pred, dec_2_pred, dec_3_pred

    def set_training(self, flag):
        if flag:
            self.train()
        else:
            self.eval()

        self.training = flag

    def _init_weights(self):
        pretrained_modules = self.encoder.modules()

        not_pretrained_modules = [
            self.middle_conv,
            self.class_conv,
            self.dec_3,
            self.dec_3_pred,
            self.dec_2,
            self.dec_2_pred,
            self.dec_1,
            self.dec_1_pred,
            self.dec_0,
            self.dec_0_pred,
            self.dec_final_0,
            self.final,
        ]

        not_pretrained_modules = chain(*[module.modules() for module in not_pretrained_modules])

        if not self.pretrained:
            self._init_modules(pretrained_modules)

        self._init_modules(not_pretrained_modules)

    @staticmethod
    def _init_modules(modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


if __name__ == '__main__':
    segmentor = SawSeenNet(base_channels=64, pretrained=True)
    print(segmentor)
    pic = torch.randn(1, 3, 128, 128)
    result, class_pred, result_64, result_32, result_16, result_8 = segmentor(pic)
    print(result.size(), class_pred.size(), result_64.size(),
          result_32.size(), result_16.size(), result_8.size())
