import torch
import torch.nn as nn
from collections import OrderedDict

def conv2d(filter_in, filter_out, kernel_size, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))

#### Related Attention based on Memory Enhancement
class RelatedAttenMemEnhanceAggregation(nn.Module):
    def __init__(self, input_img_num, output_channels=32, input_mode='GRG', mem_x_channels=4, mem_queue_length=3):
        super(RelatedAttenMemEnhanceAggregation, self).__init__()
        # input_mode: "RGB" or "GRG". "RGB" means all the image is rgb mode. "GRG" means that the middle image remains RGB,
        # and the others will be coverted to gray. 
        if input_mode == "RGB":
            input_channels = input_img_num * 3
        elif input_mode == "GRG":
            input_channels = input_img_num + 2
        else:
            raise print("input_mode error!")
        input_channels = input_channels + mem_queue_length*mem_x_channels

        self.conv1 = conv2d(input_channels, input_channels, 3)
        self.conv2 = conv2d(input_channels, input_channels, 3)
        self.conv3 = conv2d(input_channels, output_channels, 3)


        self.conv4 = conv2d(output_channels, mem_x_channels, 3)
        

    def forward(self, x, mem_queue_x):
        x = torch.cat([x,mem_queue_x],axis=1)
        out1 = self.conv1(x)
        F1 = self.conv2(out1)
        F2 = torch.sigmoid(F1)
        F3 = out1 * F2
        out2 = out1 + F3
        out = self.conv3(out2)
        mem_x = self.conv4(out)

        return out, mem_x