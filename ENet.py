import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
from collections import OrderedDict


class ENet(nn.Module):
    def __init__(self, n_classes=21, model_type='encoder_decoder'):
        super(ENet, self).__init__()
        self.model_type = model_type

        self.initial = nn.Sequential(OrderedDict([]))
        self.bottleneck1_0 = nn.Sequential(OrderedDict([]))
        self.bottleneck1 = nn.Sequential(OrderedDict([]))
        self.bottleneck2_0 = nn.Sequential(OrderedDict([]))
        self.bottleneck2 = nn.Sequential(OrderedDict([]))
        self.bottleneck3 = nn.Sequential(OrderedDict([]))
        self.bottleneck4_0 = nn.Sequential(OrderedDict([]))
        self.bottleneck4 = nn.Sequential(OrderedDict([]))
        self.bottleneck5_0 = nn.Sequential(OrderedDict([]))
        self.bottleneck5 = nn.Sequential(OrderedDict([]))

        # initial
        self.initial.add_module('initial ', InitBlock())

        # bottleneck 1.0
        self.bottleneck1_0.add_module('bn1.0',
                                      Bottleneck(stage=1, idx_bn=0,
                                                 in_size=16, mid_size=16,
                                                 out_size=64, drop_p=0.01,
                                                 layer_type='downsampling'))
        # bottleneck 1
        for i in xrange(1, 5):
            self.bottleneck1.add_module('bn1.%d' % i,
                                        Bottleneck(stage=1, idx_bn=i,
                                                   in_size=64, mid_size=16,
                                                   out_size=64, drop_p=0.01,
                                                   layer_type='regular'))
        # bottleneck 2.0
        self.bottleneck2_0.add_module('bn2.0',
                                      Bottleneck(stage=2, idx_bn=0,
                                                 in_size=64, mid_size=32,
                                                 out_size=128, drop_p=0.1,
                                                 layer_type='downsampling'))

        # bottleneck 2 & 3
        for j in xrange(2, 4):
            if j == 2:
                module = self.bottleneck2
            else:
                module = self.bottleneck3
            module.add_module('bn%d.1' % j,
                              Bottleneck(stage=j, idx_bn=1,
                                         in_size=128, mid_size=32,
                                         out_size=128, drop_p=0.1,
                                         layer_type='regular'))
            module.add_module('bn%d.2' % j,
                              Bottleneck(stage=j, idx_bn=2,
                                         in_size=128, mid_size=32,
                                         out_size=128, drop_p=0.1,
                                         layer_type='dilated', extra_param=2))
            module.add_module('bn%d.3' % j,
                              Bottleneck(stage=j, idx_bn=3,
                                         in_size=128, mid_size=32,
                                         out_size=128, drop_p=0.1,
                                         layer_type='asymm', extra_param=5))
            module.add_module('bn%d.4' % j,
                              Bottleneck(stage=j, idx_bn=4,
                                         in_size=128, mid_size=32,
                                         out_size=128, drop_p=0.1,
                                         layer_type='dilated', extra_param=4))
            module.add_module('bn%d.5' % j,
                              Bottleneck(stage=j, idx_bn=5,
                                         in_size=128, mid_size=32,
                                         out_size=128, drop_p=0.1,
                                         layer_type='regular'))
            module.add_module('bn%d.6' % j,
                              Bottleneck(stage=j, idx_bn=6,
                                         in_size=128, mid_size=32,
                                         out_size=128, drop_p=0.1,
                                         layer_type='dilated', extra_param=8))
            module.add_module('bn%d.7' % j,
                              Bottleneck(stage=j, idx_bn=7,
                                         in_size=128, mid_size=32,
                                         out_size=128, drop_p=0.1,
                                         layer_type='asymm', extra_param=5))
            module.add_module('bn%d.8' % j,
                              Bottleneck(stage=j, idx_bn=8,
                                         in_size=128, mid_size=32,
                                         out_size=128, drop_p=0.1,
                                         layer_type='dilated', extra_param=16))

        # bottleneck 4.0
        self.bottleneck4_0 = Bottleneck(stage=4, idx_bn=0, in_size=128,
                                        mid_size=32, out_size=64, drop_p=0.1,
                                        layer_type='upsampling')

        # bottleneck 4
        for i in xrange(1, 3):
            self.bottleneck4.add_module('bn4.%d' % i,
                                        Bottleneck(stage=4, idx_bn=i,
                                                   in_size=64, mid_size=16,
                                                   out_size=64, drop_p=0.01,
                                                   layer_type='regular'))

        # bottleneck 5.0
        self.bottleneck5_0 = Bottleneck(stage=5, idx_bn=0, in_size=64,
                                        mid_size=32, out_size=16, drop_p=0.1,
                                        layer_type='upsampling')

        self.bottleneck5.add_module('bn5.1',
                                    Bottleneck(stage=5, idx_bn=1,
                                               in_size=16, mid_size=16,
                                               out_size=16, drop_p=0.01,
                                               layer_type='regular'))

        if model_type == 'encoder':
            self.deconv = nn.ConvTranspose2d(128, n_classes,
                                             kernel_size=1, stride=1)
        elif model_type == 'encoder_decoder':
            self.deconv = nn.ConvTranspose2d(16, n_classes,
                                             kernel_size=2, stride=2)
        else:
            raise AttributeError('model_type: encoder or encoder_decoder')

    def forward(self, inputs):

        x = self.initial(inputs)
        x, pool1 = self.bottleneck1_0(x)
        x = self.bottleneck1(x)
        x, pool2 = self.bottleneck2_0(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        if self.model_type == 'encoder':
            x = self.deconv(x)
        elif self.model_type == 'encoder_decoder':
            x = self.bottleneck4_0(x, pool2)
            x = self.bottleneck4(x)
            x = self.bottleneck5_0(x, pool1)
            x = self.bottleneck5(x)
            x = self.deconv(x)
        else:
            raise AttributeError('model_type: encoder or encoder_decoder')

        return x


class InitBlock(nn.Module):
    def __init__(self, in_size=3, out_size=13):
        super(InitBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, 3, stride=2,
                               padding=1, bias=False)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.prelu1 = nn.PReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(x)
        ca = torch.cat([c1, p1], 1)
        ca = self.bn1(ca)
        ca = self.prelu1(ca)
        return ca


class Bottleneck(nn.Module):
    def __init__(self, stage, idx_bn, in_size, mid_size,
                 out_size, drop_p, layer_type, extra_param=2):
        super(Bottleneck, self).__init__()
        self.layer_type = layer_type

        self.bnbranch = nn.Sequential(OrderedDict([]))
        self.mainbranch = nn.Sequential(OrderedDict([]))
        # 1st module
        # bn: BatchNorm
        if layer_type == 'downsampling':
            self.bnbranch.add_module('conv %d_%d_%d' % (stage, idx_bn, 0),
                                     nn.Conv2d(in_size, mid_size, 2,
                                               stride=2, bias=False))
        else:
            self.bnbranch.add_module('conv %d_%d_%d' % (stage, idx_bn, 0),
                                     nn.Conv2d(in_size, mid_size, 1,
                                               stride=1, bias=False))
        self.bnbranch.add_module('bn %d_%d_%d' % (stage, idx_bn, 0),
                                 nn.BatchNorm2d(mid_size))
        self.bnbranch.add_module('prelu %d_%d_%d' % (stage, idx_bn, 0),
                                 nn.LeakyReLU())

        # 2nd module
        if layer_type == 'dilated':
            self.bnbranch.add_module('conv %d_%d_%d' % (stage, idx_bn, 1),
                                     nn.Conv2d(mid_size, mid_size, 3, stride=1,
                                               padding=extra_param, bias=False,
                                               dilation=extra_param))
        elif layer_type == 'asymm':
            self.bnbranch.add_module('conv %d_%d_%d _a' % (stage, idx_bn, 1),
                                     nn.Conv2d(mid_size, mid_size, stride=1,
                                               kernel_size=(extra_param, 1),
                                               padding=1, bias=False,))
            self.bnbranch.add_module('conv %d_%d_%d _' % (stage, idx_bn, 1),
                                     nn.Conv2d(mid_size, mid_size, stride=1,
                                               kernel_size=(1, extra_param),
                                               padding=1, bias=False))
        elif layer_type == 'upsampling':
            self.bnbranch.add_module('deconv %d_%d_%d' % (stage, idx_bn, 1),
                                     nn.ConvTranspose2d(mid_size, mid_size, 2,
                                                        stride=2, bias=False))
        else:
            self.bnbranch.add_module('conv %d_%d_%d' % (stage, idx_bn, 1),
                                     nn.Conv2d(mid_size, mid_size, 3,
                                               stride=1, bias=False))

        self.bnbranch.add_module('bn %d_%d_%d' % (stage, idx_bn, 1),
                                 nn.BatchNorm2d(mid_size))
        self.bnbranch.add_module('prelu %d_%d_%d' % (stage, idx_bn, 1),
                                 nn.LeakyReLU())

        # 3rd module
        self.bnbranch.add_module('conv %d_%d_%d' % (stage, idx_bn, 2),
                                 nn.Conv2d(mid_size, out_size, 1,
                                           stride=1, bias=False))
        self.bnbranch.add_module('bn %d_%d_%d' % (stage, idx_bn, 2),
                                 nn.BatchNorm2d(out_size))

        # 4th module
        self.bnbranch.add_module('drop %d_%d_%d' % (stage, idx_bn, 3),
                                 nn.Dropout2d(p=drop_p))

        # main branch
        if layer_type == 'downsampling':
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.mainbranch.add_module('conv %d_%d_%d' % (stage, idx_bn, 4),
                                       nn.Conv2d(in_size, out_size, 1,
                                                 stride=1, bias=False))
            self.mainbranch.add_module('bn %d_%d_%d' % (stage, idx_bn, 4),
                                       nn.BatchNorm2d(out_size))

        elif layer_type == 'upsampling':
            self.mainbranch.add_module('conv %d_%d_%d' % (stage, idx_bn, 4),
                                       nn.Conv2d(in_size, out_size, 1,
                                                 stride=1, bias=False))
            self.mainbranch.add_module('bn %d_%d_%d' % (stage, idx_bn, 4),
                                       nn.BatchNorm2d(out_size))

        self.prelu4 = nn.LeakyReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs1, inputs2=Variable(torch.Tensor([0, 0, 0]),
                                                requires_grad=True)):
        bb = self.bnbranch(inputs1)

        if self.layer_type == 'downsampling':
            poolmask_by_down = self.pool4(inputs1)
            mb = self.mainbranch(poolmask_by_down)
            x = bb + mb
            x = self.prelu4(x)
            return x, poolmask_by_down
        elif self.layer_type == 'upsampling':
            mb = self.mainbranch(inputs1)
            mb = mb+inputs2
            mb = F.upsample(mb, bb.size()[2:], scale_factor=2, mode='bilinear')
        else:
            mb = inputs1

        x = bb + mb
        x = self.prelu4(x)
        return x
