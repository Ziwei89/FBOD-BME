import torch.nn as nn
from collections import OrderedDict
import sys
sys.path.append("..")
from .feature_aggregation.feature_aggregation import ImagesAggregation
from .feature_extraction.feature_extraction import FeatureExtraction

def conv2d(filter_in, filter_out, kernel_size, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))


#---------------------------------------------------#
#   output
#---------------------------------------------------#
def FBODetection_head(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 3),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m

#---------------------------------------------------#
#   FBOInferenceNet
#---------------------------------------------------#
class FBODInferenceBody(nn.Module):
    def __init__(self, input_img_num=5, aggregation_output_channels=32, input_mode="GRG", mem_x_channels=4, mem_queue_length=3, ### Aggreagation parameters.
                       backbone_name="cspdarknet53", fusion_method="concat"): ### Extract parameters. input_channels equal to aggregation_output_channels.
        super(FBODInferenceBody, self).__init__()
        """
        input_mode:         "RGB" or "GRG". "RGB" means all the image is rgb mode. "GRG" means that the middle image remains RGB,
                             and the others will be coverted to gray.
        backbone_name:      "cspdarknet53" or "mobilenetv2".
        fusion_method:      "concat" or "scm".
        """
        self.aggregate_features = ImagesAggregation(input_img_num=input_img_num, aggregation_output_channels=aggregation_output_channels,
                                                    input_mode=input_mode, mem_x_channels=mem_x_channels, mem_queue_length=mem_queue_length)
        ### The input_channels of feature extraction net is equal to aggregation_output_channels.
        self.extract_features = FeatureExtraction(backbone_name=backbone_name, input_channels=aggregation_output_channels, fusion_method=fusion_method)

        self.FBODetection_head_conf = FBODetection_head([32, 1],32)
        self.FBODetection_head_pos = FBODetection_head([32, 4],32)


    def forward(self, x, mem_queue_x=None, mem_queue_input=False):

        out0, mem_x = self.aggregate_features(x, mem_queue_x, mem_queue_input) # output channels = 32
        out1 = self.extract_features(out0) # output channels = 32

        conf = self.FBODetection_head_conf(out1) # output channels = 1
        pos = self.FBODetection_head_pos(out1) # output channels = 4

        return conf, pos, mem_x 