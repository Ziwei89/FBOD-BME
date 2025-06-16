import os
import xml.etree.ElementTree as ET
import argparse

num_to_chinese_c_dic = {1:"one", 3:"three", 5:"five", 7:"seven", 9:"nine", 11:"eleven"}
def difficulty_class_to_conf(difficulty_class_level):
    return (0.875 - difficulty_class_level/4)
classes=['n01503061']
def convert_annotation(annotation_file, list_file):
    in_file = open(annotation_file, encoding='utf-8')
    tree=ET.parse(in_file)
    root = tree.getroot()

    if root.find('object')!=None:
        for obj in root.iter('object'):
            difficult = 0 
            if obj.find('difficult')!=None:
                difficult = obj.find('difficult').text
            obj_conf = difficulty_class_to_conf(int(difficult))
                
            cls = obj.find('name').text
            if cls not in classes:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
            list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id) + ',' + str(obj_conf))
    else:
        list_file.write(" " + "None")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_path', default="../dataset/FBD-SV-2024/", type=str,
                        help='data_root_path: The path of the dataset.')
    parser.add_argument('--input_img_num', default=5, type=int,
                        help='input_img_num: The continous video frames, input to the model')
    parser.add_argument('--img_ext', default=".JPEG", type=str,
                        help='img_ext: The extension name of the image')
    args = parser.parse_args()

    train_img_label_txt_files_path = "../TrainFramework/dataloader/train_video_img_label_txt_files/"
    val_img_label_txt_files_path = "../TrainFramework/dataloader/val_video_img_label_txt_files/"

    os.makedirs(train_img_label_txt_files_path, exist_ok=True)
    os.makedirs(val_img_label_txt_files_path, exist_ok=True)

    img_label_txt_files_paths = [train_img_label_txt_files_path, val_img_label_txt_files_path]

    train_label_path = args.data_root_path + "VID/labels/train/"
    val_label_path = args.data_root_path + "VID/labels/val/"
    label_pathes = [train_label_path, val_label_path]

    train_image_path = args.data_root_path + "VID/images/train/"
    val_image_path = args.data_root_path + "VID/images/val/"
    image_pathes = [train_image_path, val_image_path]

    for label_path, image_path, img_label_txt_files_path in zip(label_pathes, image_pathes, img_label_txt_files_paths):
        video_names = os.listdir(label_path)
        for video_name in video_names:
            img_label_txt_file = img_label_txt_files_path + video_name + "_img_label_" + num_to_chinese_c_dic[args.input_img_num] + "_continuous_difficulty.txt"
            list_file = open(img_label_txt_file, 'w')
            label_files = os.listdir(label_path + video_name + "/")
            label_file_num = len(label_files)
            for i in range(label_file_num):
                num_str = "%06d" % int(i)
                label_name = num_str + ".xml"
                image_name = num_str + args.img_ext
                if not os.path.exists(label_path + video_name + "/" + label_name):
                    raise print("Error, No file:", label_path + video_name + "/" + label_name)
                if not os.path.exists(image_path + video_name + "/" + image_name):
                    raise print("Error, No file:", image_path + video_name + "/" + image_name)
                first_image_num_str = "%06d" % int(i-int(args.input_img_num/2))
                first_image_name = first_image_num_str + args.img_ext
                print(first_image_name)
                list_file.write(first_image_name)
                lable_str = convert_annotation(label_path + video_name + "/" + label_name, list_file)
                list_file.write("\n")
            list_file.close()