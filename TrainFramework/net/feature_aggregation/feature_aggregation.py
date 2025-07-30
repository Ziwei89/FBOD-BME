import torch.nn as nn
import sys
sys.path.append("..")

from .relatedatten_memenhance_method import RelatedAttenMemEnhanceAggregation



class ImagesAggregation(nn.Module):
    def __init__(self, input_img_num=5, aggregation_output_channels=32, input_mode="GRG", mem_x_channels=4, mem_queue_length=3):
        super(ImagesAggregation, self).__init__()
        # input_mode: "RGB" or "GRG". "RGB" means all the image is rgb mode. "GRG" means that the middle image remains RGB,
        # and the others will be coverted to gray.
        self.images_fusion_module_mem_enhance = RelatedAttenMemEnhanceAggregation(input_img_num=input_img_num, output_channels=aggregation_output_channels, input_mode=input_mode, mem_x_channels=mem_x_channels, mem_queue_length=mem_queue_length)


    def forward(self, x, mem_queue_x):#
        fusion_img, mem_x = self.images_fusion_module_mem_enhance(x, mem_queue_x)
        return fusion_img, mem_x