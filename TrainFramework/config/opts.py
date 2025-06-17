import argparse

class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        
        self.parser.add_argument('--model_input_size', default="384_672", type=str,
                            help='model_input_size: The model input shape h_w')
        
        self.parser.add_argument('--input_img_num', default=5, type=int,
                            help='input_img_num: The continous video frames, input to the model')
        
        self.parser.add_argument('--aggregation_output_channels', default=16, type=int,
                            help='aggregation_output_channels: The output channels of the aggregation module')
        
        self.parser.add_argument('--input_mode', default="RGB", type=str,
                            help='input_mode: "RGB" or "GRG". "RGB": mean RGB ..RGB, "GRG": mean GRAY ..RGB.. GRAY')
        
        self.parser.add_argument('--mem_x_channels', default=4, type=int,
                            help='mem_x_channels: The channels of mem_x')
        
        self.parser.add_argument('--mem_queue_length', default=3, type=int,
                            help='mem_queue_length: The length of the mem queue')
        
        self.parser.add_argument('--backbone_name', default='cspdarknet53', type=str,
                            help='backbone_name: cspdarknet53 or CustomNet_model')
        
        self.parser.add_argument('--fusion_method', default='concat', type=str,
                            help='fusion_method: concat or scm')
        
        self.parser.add_argument('--assign_method', default='guassian_assign', type=str,
                            help='assign_method: The label assign method. binary_assign, guassian_assign, auto_assign, or auto_guassian_assign')
        
        self.parser.add_argument('--Add_name', default='0816_1', type=str,
                            help='Add_name: add name to logs and pic')
        ######### for train
        self.parser.add_argument('--Batch_size', default=8, type=int,
                            help='Batch_size: The size of batch.')
        
        self.parser.add_argument('--data_augmentation', default=True, type=bool,
                            help='data_augmentation: Determin whether to augmentate the dataset.')
        
        self.parser.add_argument('--pretrain_model_path', default="logs/non.pth", type=str,
                            help='pretrain_model_path: the pretrain model to speed train.')
        
        self.parser.add_argument('--data_root_path', default="../../dataset/FBD-SV-2024/", type=str,
                            help='data_root_path: The path of the dataset.')
        
        self.parser.add_argument('--video_train_annotation_path', default="./dataloader/train_video_img_label_txt_files/", type=str,
                            help='video_train_annotation_path: The path of the train annotation files.')
        
        self.parser.add_argument('--video_val_annotation_path', default="./dataloader/val_video_img_label_txt_files/", type=str,
                            help='video_val_annotation_path: The path of the val annotation files.')
        
        self.parser.add_argument('--start_Epoch', default=0, type=int,
                            help='start_Epoch: the start epoch.')
        
        self.parser.add_argument('--middle_Epoch', default=30, type=int,
                            help='start_Epoch: the start epoch.')
        
        self.parser.add_argument('--end_Epoch', default=100, type=int,
                            help='end_Epoch: the end epoch.')

        ######### for test
        self.parser.add_argument('--model_name', default="FB_object_detect_model.pth", type=str,
                            help='model_name: The model name for loade.')
        
        self.parser.add_argument('--video_path', default="../../FBOD-BSPL/dataset/val/videos/", type=str,
                            help='video_path: The video path.')
        
        self.parser.add_argument('--video_name', default="bird_2.mp4", type=str,
                            help='video_name: The vido name for testing.')
        
        
    def parse(self):
        opt = self.parser.parse_args()
        return opt