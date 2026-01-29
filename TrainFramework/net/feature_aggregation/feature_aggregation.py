import torch.nn as nn
import sys
sys.path.append("..")

from .relatedatten_memenhance_method import RelatedAttenMemEnhanceAggregation
from .relatedatten_method import RelatedAttenAggregation
from .convlstm_method import ConvLSTMAggregation



class ImagesAggregation(nn.Module):
    def __init__(self, input_img_num=5, aggregation_output_channels=32, aggregation_method="relatedatten_memenhance", input_mode="GRG", mem_x_channels=4, mem_queue_length=3):
        super(ImagesAggregation, self).__init__()
        # aggregation_method: "relatedatten_memenhance" , "relatedatten" or "convlstm". "relatedatten_memenhance" means Related attention memory enhanced method, "relatedatten" means Related attention method, and "convlstm" means ConvLSTM.
        # input_mode: "RGB" or "GRG". "RGB" means all the image is rgb mode. "GRG" means that the middle image remains RGB,
        # and the others will be coverted to gray.
        self.aggregation_method = aggregation_method
        if self.aggregation_method=="relatedatten_memenhance":
            self.images_fusion_module = RelatedAttenMemEnhanceAggregation(input_img_num=input_img_num, output_channels=aggregation_output_channels, input_mode=input_mode, mem_x_channels=mem_x_channels, mem_queue_length=mem_queue_length)
        elif self.aggregation_method=="relatedatten":
            self.images_fusion_module = RelatedAttenAggregation(input_img_num=input_img_num, output_channels=aggregation_output_channels, input_mode=input_mode)
        elif self.aggregation_method=="convlstm":
            if input_mode=="GRG":
                raise("Error! When the aggregation methord is 'convlstm', the input mode must be 'RGB'.")
            self.images_fusion_module = ConvLSTMAggregation(input_img_num=input_img_num, output_channels=aggregation_output_channels)
        else:
            raise("aggregation_method error!")


    def forward(self, x, mem_queue_x=None):#
        if self.aggregation_method=="relatedatten_memenhance":
            fusion_img, mem_x = self.images_fusion_module(x, mem_queue_x)
            return fusion_img, mem_x
        else:
            fusion_img = self.images_fusion_module(x)
            return fusion_img