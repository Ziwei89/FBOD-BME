#-------------------------------------#
#       对数据集进行训练
#-------------------------------------#
import os
from config.opts import opts
import numpy as np
import time
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from net.FBODInferenceNet import FBODInferenceBody
from utils.FBODLoss import LossFunc
from FB_detector import FB_Postprocess
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.utils import FBObj
from dataloader.dataset_bbox import CustomDataset, dataset_collate
from mAP import mean_average_precision
import copy
import random
from queue import Queue
import math
from utils.common import load_model

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def iterate_n_elements(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

#---------------------------------------------------#
#   获得类
#---------------------------------------------------#
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class LablesToResults(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def covert(self, labels_list, iteration):
        label_obj_list = []
        for batch_id in range(self.batch_size):
            labels = labels_list[batch_id]
            if labels.size==0:
                continue
            image_id = self.batch_size*iteration + batch_id
            for label in labels:
                # class_id = label[4] + 1 ###Include background in this project, the label didn't include background classes.
                box = [label[i] for i in range(4)]
                label_obj_list.append(FBObj(score=1., image_id=image_id, bbox=box))
        return label_obj_list

def LablesToResults_video(labels, image_id):
    label_obj_list = []
    labels = labels[0]
    if labels.size==0:
        pass
    else:
        for label in labels:
            box = [label[i] for i in range(4)]
        label_obj_list.append(FBObj(score=1., image_id=image_id, bbox=box))
    return label_obj_list

def generate_batch_mem_queue_x(batch_size, queue_size=3, model_input_size=(384,672), cuda=True):
    batch_mem_queue_x = torch.zeros(batch_size, 4*queue_size, model_input_size[0], model_input_size[1])
    if cuda:
        batch_mem_queue_x.cuda()
    return batch_mem_queue_x

def init_mem_x_q(queue_size=3, model_input_size=(384,672), cuda=True):
    mem_x_q = Queue(maxsize=queue_size)
    for _ in range(queue_size):
        mem_x = torch.zeros(1, 4, model_input_size[0], model_input_size[1])
        if cuda:
            mem_x = mem_x.cuda()
        mem_x_q.put(mem_x)
    return mem_x_q

def init_batch_mem_x_q(batch_size, queue_size=3, model_input_size=(384,672), cuda=True):
    batch_mem_x_q = Queue(maxsize=queue_size)
    for _ in range(queue_size):
        batch_mem_x = torch.zeros(batch_size, 4, model_input_size[0], model_input_size[1])
        if cuda:
            batch_mem_x = batch_mem_x.cuda()
        batch_mem_x_q.put(batch_mem_x)
    return batch_mem_x_q

def concat_mem_x_q(mem_x_q):
    mem_x0, mem_x1, mem_x2 = mem_x_q.queue[0], mem_x_q.queue[1], mem_x_q.queue[2]
    mem_queue_x = torch.cat([mem_x0, mem_x1, mem_x2], axis=1)
    return mem_queue_x

def read_train_annotation_files(video_train_annotation_path, data_root_path, sublist_video_train_annotation_file):
    list_train_lines = []
    list_train_dataset_image_path = []
    min_video_length=1000000
    for video_train_annotation_file in sublist_video_train_annotation_file:
        with open(video_train_annotation_path + video_train_annotation_file) as f:
            train_lines = f.readlines()
            list_train_lines.append(train_lines)
            video_length = len(train_lines)
            if video_length < min_video_length:
                min_video_length = video_length
        video_name = video_train_annotation_file.split("_")[0] + "_" + video_train_annotation_file.split("_")[1]
        # print(video_name)
        train_dataset_image_path = data_root_path + "VID/images/train/" + video_name + "/"
        list_train_dataset_image_path.append(train_dataset_image_path)
    return list_train_lines, min_video_length, list_train_dataset_image_path

def train_videos(net, loss_func, epoch, end_Epoch, num_trained_video, num_train_video, min_video_length, list_train_dataloader, batch_mem_x_q, cuda):
    net.train()
    total_loss = 0
    actual_batch_size = len(list_train_dataloader)
    with tqdm(total=min_video_length,desc=f'Epoch {epoch + 1}/{end_Epoch}, trained video {num_trained_video}/{num_train_video}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batches in enumerate(zip(*list_train_dataloader)):
            if iteration >= min_video_length:
                break
            image_list = []
            targets = []
            for i in range(actual_batch_size):
                image, target = batches[i][0], batches[i][1]
                image_list.append(image)
                targets += target
            images = np.concatenate(image_list, axis=0)
            with torch.no_grad():
                if cuda:
                    images = Variable(torch.from_numpy(images)).to(torch.device('cuda:0'))
                    targets = [Variable(torch.from_numpy(fature_label)) for fature_label in targets] ## 
                else:
                    images = Variable(torch.from_numpy(images))
                    targets = [Variable(torch.from_numpy(fature_label).type(torch.FloatTensor)) for fature_label in targets] ##
                batch_mem_queue_x = concat_mem_x_q(batch_mem_x_q)
            optimizer.zero_grad()
            outputs = net(images, batch_mem_queue_x)
            _ = batch_mem_x_q.get()
            batch_mem_x = outputs[2]
            batch_mem_x_q.put(batch_mem_x)

            if loss_func.cuda == False:
                loss = loss_func(outputs.to(torch.device('cpu')), targets)
            else:
                loss = loss_func(outputs, targets)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                total_loss += loss
            pbar.set_postfix(**{'total_loss': total_loss.item() / (iteration + 1), 
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)
    return total_loss/min_video_length

def val_one_video(net, loss_func, epoch, end_Epoch, video_num, num_train_video, total_frame_length, video_length, genval, mem_x_q, cuda, detect_post_process):
    net.eval()
    val_loss = 0
    all_label_obj_list = []
    all_obj_result_list = []
    with tqdm(total=video_length,desc=f'Epoch {epoch + 1}/{end_Epoch}, video {video_num + 1}/{num_train_video}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= video_length:
                break
            images_val, targets_val = batch[0], batch[1]
            labels_list = copy.deepcopy(targets_val)
            with torch.no_grad():
                if cuda:
                    images_val = Variable(torch.from_numpy(images_val)).to(torch.device('cuda:0'))
                    targets_val = [Variable(torch.from_numpy(fature_label)) for fature_label in targets_val] ## 
                else:
                    images_val = Variable(torch.from_numpy(images_val))
                    targets_val = [Variable(torch.from_numpy(fature_label).type(torch.FloatTensor)) for fature_label in targets_val] ##
                
                mem_queue_x = concat_mem_x_q(mem_x_q)
                outputs = net(images_val, mem_queue_x)

                _ = mem_x_q.get()
                mem_x = outputs[2]
                mem_x_q.put(mem_x)

                if loss_func.cuda == False:
                    loss = loss_func(outputs.to(torch.device('cpu')), targets_val)
                else:
                    loss = loss_func(outputs, targets_val)
                val_loss += loss
                if (epoch+1) >= 30:
                    # image_id = total_frame_length + iteration
                    # label_obj_list = LablesToResults_video(labels, image_id)
                    label_obj_list = labels_to_results.covert(labels_list, total_frame_length + iteration)
                    
                    # obj_num = len(label_obj_list)
                    # for i in range(obj_num):
                    #     print("label_obj_list[{}].image_id, label_obj_list[{}].bbox:".format(i,i))
                    #     print(label_obj_list[i].image_id, label_obj_list[i].bbox)
                    all_label_obj_list += label_obj_list

                    # obj_result_list = detect_post_process.Process_video(outputs, image_id)
                    obj_result_list = detect_post_process.Process(outputs, total_frame_length + iteration)
                    # obj_num = len(obj_result_list)
                    # for i in range(obj_num):
                    #     print("obj_result_list[{}].image_id, obj_result_list[{}].bbox:".format(i,i))
                    #     print(obj_result_list[i].image_id, obj_result_list[i].bbox)
                    all_obj_result_list += obj_result_list

                pbar.set_postfix(**{'total_loss': val_loss.item() / (iteration + 1)})
                pbar.update(1)
    if (epoch+1) >= 30:
        return val_loss/video_length, all_label_obj_list, all_obj_result_list
    else:
        return val_loss/video_length, None, None

def fit_one_epoch(largest_AP_50,net,loss_func,epoch,epoch_size,epoch_size_val,gen,genval,Epoch,batch_mem_queue_x,cuda,save_model_dir,labels_to_results,detect_post_process):
    total_loss = 0
    val_loss = 0
    start_time = time.time()
    with tqdm(total=epoch_size,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            images, targets, names = batch[0], batch[1], batch[2]
            #print(images.shape) 1,7,384,672
            with torch.no_grad():
                if cuda:
                    images = Variable(torch.from_numpy(images)).to(torch.device('cuda:0'))
                    targets = [Variable(torch.from_numpy(fature_label)) for fature_label in targets] ## 
                else:
                    images = Variable(torch.from_numpy(images))
                    targets = [Variable(torch.from_numpy(fature_label).type(torch.FloatTensor)) for fature_label in targets] ##
            optimizer.zero_grad()
            if batch_mem_queue_x == None:
                outputs = net(images)
            else:
                outputs = net(images, batch_mem_queue_x)
            if loss_func.cuda == False:
                loss = loss_func(outputs, targets)
            else:
                loss = loss_func(outputs, targets)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                total_loss += loss
            waste_time = time.time() - start_time
            
            pbar.set_postfix(**{'total_loss': total_loss.item() / (iteration + 1), 
                                'lr'        : get_lr(optimizer),
                                'step/s'    : waste_time})
            pbar.update(1)

            start_time = time.time()
    net.eval()
    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        all_label_obj_list = []
        all_obj_result_list = []
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            images_val, targets_val = batch[0], batch[1]
            labels_list = copy.deepcopy(targets_val)
            with torch.no_grad():
                if cuda:
                    images_val = Variable(torch.from_numpy(images_val)).to(torch.device('cuda:0'))
                    targets_val = [Variable(torch.from_numpy(fature_label)) for fature_label in targets_val] ## 
                else:
                    images_val = Variable(torch.from_numpy(images_val))
                    targets_val = [Variable(torch.from_numpy(fature_label).type(torch.FloatTensor)) for fature_label in targets_val] ##
                optimizer.zero_grad()
                if batch_mem_queue_x == None:
                    outputs = net(images_val)
                else:
                    outputs = net(images_val, batch_mem_queue_x)

                if loss_func.cuda == False:
                    loss = loss_func(outputs, targets_val)
                else:
                    loss = loss_func(outputs, targets_val)
                val_loss += loss

                if (epoch+1) >= 30:
                    label_obj_list = labels_to_results.covert(labels_list, iteration)
                    all_label_obj_list += label_obj_list

                    obj_result_list = detect_post_process.Process(outputs, iteration)
                    all_obj_result_list += obj_result_list

            pbar.set_postfix(**{'total_loss': val_loss.item() / (iteration + 1)})
            pbar.update(1)
    net.train()
    if (epoch+1) >= 30:
        AP_50,REC_50,PRE_50=mean_average_precision(all_obj_result_list,all_label_obj_list,iou_threshold=0.5)
    else:
        AP_50,REC_50,PRE_50 = 0,0,0
    
    print('Finish Validation')
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f  || AP_50: %.4f  || REC_50: %.4f  || PRE_50: %.4f' % (total_loss/(epoch_size+1), val_loss/(epoch_size_val+1),  AP_50, REC_50, PRE_50))
    
    if (epoch+1)%10 == 0 or epoch == 0:
        if largest_AP_50 < AP_50:
            largest_AP_50 = AP_50
        print('Saving state, iter:', str(epoch+1))
        torch.save(model.state_dict(), save_model_dir + 'Epoch%d-Total_Loss%.4f-Val_Loss%.4f-AP_50_%.4f.pth'%((epoch+1),total_loss/(epoch_size+1),val_loss/(epoch_size_val+1),AP_50))
        torch.save(model.state_dict(), save_model_dir + 'FB_object_detect_model.pth')
    else:
        if largest_AP_50 < AP_50:
            largest_AP_50 = AP_50
            print('Saving state, iter:', str(epoch+1))
            torch.save(model.state_dict(), save_model_dir + 'Epoch%d-Total_Loss%.4f-Val_Loss%.4f-AP_50_%.4f.pth'%((epoch+1),total_loss/(epoch_size+1),val_loss/(epoch_size_val+1),AP_50))
            torch.save(model.state_dict(), save_model_dir + 'FB_object_detect_model.pth')
    if (epoch+1) >= 30:
        return total_loss/(epoch_size+1), val_loss/(epoch_size_val+1), largest_AP_50, AP_50
    else:
        return total_loss/(epoch_size+1), val_loss/(epoch_size_val+1), largest_AP_50, 0.80

num_to_english_c_dic = {3:"three", 5:"five", 7:"seven", 9:"nine", 11:"eleven"}

####################### Plot figure #######################################
x_epoch = []
record_loss = {'train_loss':[], 'test_loss':[]}
fig = plt.figure()

ax0 = fig.add_subplot(111, title="Train the FB_object_detect model")
ax0.set_ylabel('loss')
ax0.set_xlabel('Epochs')

def draw_curve_loss(epoch, train_loss, test_loss, pic_name):
    global record_loss
    record_loss['train_loss'].append(train_loss)
    record_loss['test_loss'].append(test_loss)

    x_epoch.append(int(epoch))
    ax0.plot(x_epoch, record_loss['train_loss'], 'b', label='train')
    ax0.plot(x_epoch, record_loss['test_loss'], 'r', label='val')
    if epoch == 1:
        ax0.legend()
    fig.savefig(pic_name)
########============================================================########
x_ap50_epoch = []
record_ap50 = {'AP_50':[]}
fig_ap50 = plt.figure()

ax1 = fig_ap50.add_subplot(111, title="Train the FB_object_detect model")
ax1.set_ylabel('ap_50')
ax1.set_xlabel('Epochs')

def draw_curve_ap50(epoch, ap_50, pic_name):
    global record_ap50
    record_ap50['AP_50'].append(ap_50)

    x_ap50_epoch.append(int(epoch))
    ax1.plot(x_ap50_epoch, record_ap50['AP_50'], 'g', label='AP_50')
    if epoch == 30:
        ax1.legend()
    fig_ap50.savefig(pic_name)
#############################################################################

if __name__ == "__main__":

    opt = opts().parse()
    # assign_method: The label assign method. binary_assign, guassian_assign, auto_assign or auto_guassian_assign
    if opt.assign_method == "auto_assign":
        abbr_assign_method = "aa"
    elif opt.assign_method == "auto_guassian_assign":
        abbr_assign_method = "aga"
    else:
        raise print("Error! assign_method error.")
    
    save_model_dir = "logs/" + num_to_english_c_dic[opt.input_img_num] + "/" + opt.model_input_size + "/" + opt.aggregation_method + "_" + opt.input_mode \
                             + "_" + opt.backbone_name + "_" + opt.fusion_method + "_" + abbr_assign_method + "_"  + opt.Add_name + "/"
    os.makedirs(save_model_dir, exist_ok=True)

    ############### For log figure ################
    log_pic_name_loss = save_model_dir + "loss.jpg"
    log_pic_name_ap50 = save_model_dir + "ap50.jpg"
    ################################################
    config_txt = save_model_dir + "config.txt"
    if os.path.exists(config_txt):
        pass
    else:
        config_txt_file = open(config_txt, 'w')
        config_txt_file.write("Aggregaton method: " + opt.aggregation_method + "\n")
        config_txt_file.write("Input mode: " + opt.input_mode + "\n")
        config_txt_file.write("Data root path: " + opt.data_root_path + "\n")
        config_txt_file.write("Backbone name: " + opt.backbone_name + "\n")
        config_txt_file.write("Fusion method: " + opt.fusion_method + "\n")
        config_txt_file.write("Assign method: " + opt.assign_method + "\n")
        config_txt_file.write("Pretrain model: " + opt.pretrain_model_path + "\n")
        config_txt_file.write("Aggregation output channels: " + str(opt.aggregation_output_channels) + "\n")
        config_txt_file.write("Memory queue length: " + str(opt.mem_queue_length) + "\n")
        config_txt_file.write("Scale factor: " + str(opt.scale_factor) + "\n")
        config_txt_file.write("Batch size: " + str(opt.Batch_size) + "\n")
        config_txt_file.write("Data augmentation: " + str(opt.data_augmentation) + "\n")
        config_txt_file.write("Learn rate: " + str(opt.lr) + "\n")
        config_txt_file.write("Start Epoch: " + str(opt.start_Epoch) + "\n")
        config_txt_file.write("Middle Epoch: " + str(opt.middle_Epoch) + "\n")
        config_txt_file.write("End Epoch: " + str(opt.end_Epoch) + "\n")
        config_txt_file.close()

    #-------------------------------#
    #-------------------------------#
    model_input_size = (int(opt.model_input_size.split("_")[0]), int(opt.model_input_size.split("_")[1])) # H,W
    
    Cuda = True

    train_annotation_path = "./dataloader/" + "img_label_" + num_to_english_c_dic[opt.input_img_num] + "_continuous_difficulty_train.txt"
    train_dataset_image_path = opt.data_root_path + "images/train/"
    
    val_annotation_path = "./dataloader/" + "img_label_" + num_to_english_c_dic[opt.input_img_num] + "_continuous_difficulty_val.txt"
    val_dataset_image_path = opt.data_root_path + "images/val/"
    #-------------------------------#
    # 
    #-------------------------------#
    classes_path = 'model_data/classes.txt'   
    class_names = get_classes(classes_path)
    num_classes = len(class_names) + 1 #### Include background
    
    # create model
    ### FBODInferenceBody parameters:
    ### input_img_num=5, aggregation_output_channels=32, aggregation_method="relatedatten_memenhance", input_mode="GRG", mem_x_channels=4, mem_queue_length=3, ### Aggreagation parameters.
    ### backbone_name="cspdarknet53", fusion_method="concat" ### Extract parameters. input_channels equal to aggregation_output_channels.
    model = FBODInferenceBody(input_img_num=opt.input_img_num, aggregation_output_channels=opt.aggregation_output_channels, aggregation_method=opt.aggregation_method,
                              input_mode=opt.input_mode,mem_x_channels=opt.mem_x_channels,mem_queue_length=opt.mem_queue_length, backbone_name=opt.backbone_name, fusion_method=opt.fusion_method)

    #-------------------------------------------#
    #   load model
    #-------------------------------------------#
    if os.path.exists(opt.pretrain_model_path):
        print('Loading weights into state dict...')
        if Cuda:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device('cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(opt.pretrain_model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print('Finished loading pretrained model!')
    else:
        print('Train the model from scratch!')

    net = model.train()

    if Cuda:
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
        net = net.cuda()

    # 建立loss函数
    # dynamic label assign, so the gettargets is ture.
    loss_func = LossFunc(num_classes=num_classes, model_input_size=(model_input_size[1], model_input_size[0]),
                         scale=opt.scale_factor, cuda=Cuda, gettargets=True, assign_method=opt.assign_method)

    # For calculating the AP50
    detect_post_process = FB_Postprocess(batch_size=opt.Batch_size, model_input_size=model_input_size, scale=opt.scale_factor)
    labels_to_results = LablesToResults(batch_size=opt.Batch_size)


    with open(train_annotation_path) as f:
        train_lines = f.readlines()
        num_train = len(train_lines)
    with open(val_annotation_path) as f:
        val_lines = f.readlines()
        num_val = len(val_lines)
    
    #------------------------------------------------------#
    #------------------------------------------------------#
    

    start_Epoch = opt.start_Epoch
    middle_Epoch = opt.middle_Epoch
    end_Epoch = opt.end_Epoch

    lr = opt.lr * (0.95**start_Epoch)
    Batch_size = opt.Batch_size

    optimizer = optim.Adam(net.parameters(),lr,weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.95)
    
    train_data = CustomDataset(train_lines, (model_input_size[1], model_input_size[0]), image_path=train_dataset_image_path,
                               input_mode=opt.input_mode, continues_num=opt.input_img_num, data_augmentation=opt.data_augmentation)
    train_dataloader = DataLoader(train_data, batch_size=Batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=dataset_collate)
    # train_dataloader = DataLoader(train_data, batch_size=Batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    val_data = CustomDataset(val_lines, (model_input_size[1], model_input_size[0]), image_path=val_dataset_image_path,
                             input_mode=opt.input_mode, continues_num=opt.input_img_num, data_augmentation=False)
    val_dataloader = DataLoader(val_data, batch_size=Batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=dataset_collate)
    # val_dataloader = DataLoader(val_data, batch_size=Batch_size, shuffle=True, num_workers=4, pin_memory=True)


    epoch_size = max(1, num_train//Batch_size)
    epoch_size_val = num_val//Batch_size

    if opt.aggregation_method=="relatedatten_memenhance":
        batch_mem_queue_x = generate_batch_mem_queue_x(batch_size=Batch_size, queue_size=3, model_input_size=(model_input_size[0], model_input_size[1]), cuda=Cuda)
    else:
        batch_mem_queue_x=None
    largest_AP_50=0
    for epoch in range(start_Epoch,middle_Epoch):
        train_loss, val_loss,largest_AP_50_record, AP_50 = fit_one_epoch(largest_AP_50,net,loss_func,epoch,epoch_size,epoch_size_val,train_dataloader,val_dataloader,middle_Epoch,batch_mem_queue_x,Cuda,save_model_dir, labels_to_results=labels_to_results, detect_post_process=detect_post_process)
        largest_AP_50 = largest_AP_50_record
        if (epoch+1)>=2:
            draw_curve_loss(epoch+1, train_loss.item(), val_loss.item(), log_pic_name_loss)
        if (epoch+1)>=30:
            draw_curve_ap50(epoch+1, AP_50, log_pic_name_ap50)
        lr_scheduler.step()
    model_names = os.listdir(save_model_dir)
    for model_name in model_names:
        if "Epoch" in model_name:
            pass
        else:
            continue
        if "%.4f"%(largest_AP_50) == model_name.split("AP_50_")[1].split(".pth")[0]:
            largest_AP_50_model_name = model_name
            break
    # To load the largest_AP_50 model
    net = load_model(model, save_model_dir + largest_AP_50_model_name, cuda=Cuda)
    net = net.train()


    if opt.aggregation_method=="relatedatten_memenhance":
        lr = opt.lr * (0.95**middle_Epoch)
        optimizer = optim.Adam(net.parameters(),lr,weight_decay=5e-4)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.95)

        video_train_annotation_files = os.listdir(opt.video_train_annotation_path)
        num_train_video = len(video_train_annotation_files)
        video_val_annotation_files = os.listdir(opt.video_val_annotation_path)
        num_val_video = len(video_val_annotation_files)

        detect_post_process = FB_Postprocess(batch_size=1, model_input_size=model_input_size, scale=opt.scale_factor)
        labels_to_results = LablesToResults(batch_size=1)

        for epoch in range(middle_Epoch, end_Epoch):
            random.shuffle(video_train_annotation_files)
            train_loss = 0
            num_trained_video = 0
            for sublist_video_train_annotation_file in iterate_n_elements(video_train_annotation_files, Batch_size):
                actual_batch_size = len(sublist_video_train_annotation_file)
                num_trained_video += actual_batch_size
                list_train_lines, min_video_length, list_train_dataset_image_path= read_train_annotation_files(opt.video_train_annotation_path, opt.data_root_path, sublist_video_train_annotation_file)
                
                list_train_dataloader = []
                for i in range(actual_batch_size):
                    train_data = CustomDataset(list_train_lines[i], (model_input_size[1], model_input_size[0]), image_path=list_train_dataset_image_path[i],
                                            input_mode=opt.input_mode, continues_num=opt.input_img_num, data_augmentation=opt.data_augmentation)
                    train_dataloader = DataLoader(train_data, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, collate_fn=dataset_collate)
                    list_train_dataloader.append(train_dataloader)
                
                batch_mem_x_q = init_batch_mem_x_q(actual_batch_size, queue_size=3, model_input_size=(model_input_size[0], model_input_size[1]), cuda=Cuda)
                train_loss_ = train_videos(net, loss_func, epoch, end_Epoch, num_trained_video, num_train_video, min_video_length, list_train_dataloader, batch_mem_x_q, Cuda)
                train_loss += train_loss_
            train_loss = train_loss/math.ceil(num_train_video/actual_batch_size)

            random.shuffle(video_val_annotation_files)
            val_loss = 0
            all_label_obj_list = []
            all_obj_result_list = []
            total_frame_length = 0
            for video_num, video_val_annotation_file in enumerate(video_val_annotation_files):
                with open(opt.video_val_annotation_path + video_val_annotation_file) as f:
                    val_lines = f.readlines()
                    video_length = len(val_lines)
                video_name = video_val_annotation_file.split("_")[0] + "_" + video_val_annotation_file.split("_")[1]
                val_dataset_image_path = opt.data_root_path + "VID/images/val/" + video_name + "/"
                val_data = CustomDataset(val_lines, (model_input_size[1], model_input_size[0]), image_path=val_dataset_image_path,
                                        input_mode=opt.input_mode, continues_num=opt.input_img_num, data_augmentation=False)
                val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, collate_fn=dataset_collate)
                mem_x_q = init_mem_x_q(queue_size=3, model_input_size=(model_input_size[0], model_input_size[1]), cuda=Cuda)
                val_loss_temp, all_label_obj_list_temp, all_obj_result_list_temp = val_one_video(net,loss_func, epoch, end_Epoch, video_num, num_val_video, total_frame_length, video_length, val_dataloader, mem_x_q, Cuda, detect_post_process=detect_post_process)
                total_frame_length += video_length

                val_loss = val_loss + val_loss_temp
                if (epoch+1) >= 30:
                    all_label_obj_list = all_label_obj_list + all_label_obj_list_temp
                    all_obj_result_list = all_obj_result_list + all_obj_result_list_temp
            val_loss /= num_val_video
            if (epoch+1) >= 30:
                AP_50,REC_50,PRE_50=mean_average_precision(all_obj_result_list,all_label_obj_list,iou_threshold=0.5)
            else:
                AP_50,REC_50,PRE_50=0,0,0

            print('Epoch:'+ str(epoch+1) + '/' + str(end_Epoch))
            print('Total Loss: %.4f || Val Loss: %.4f  || AP_50: %.4f  || REC_50: %.4f  || PRE_50: %.4f' % (train_loss, val_loss, AP_50, REC_50, PRE_50))
            
            if largest_AP_50 < AP_50:
                largest_AP_50 = AP_50
                largest_AP_50_model_name = 'Epoch%d-Total_Loss%.4f-Val_Loss%.4f-AP_50_%.4f.pth'%((epoch+1),train_loss,val_loss,AP_50)
                print('Saving state, iter:', str(epoch+1))
                torch.save(model.state_dict(), save_model_dir + largest_AP_50_model_name)
                torch.save(model.state_dict(), save_model_dir + 'FB_object_detect_model.pth')
            else:
                if (epoch+1)%10 == 0:
                    print('Saving state, iter:', str(epoch+1))
                    torch.save(model.state_dict(), save_model_dir + 'Epoch%d-Total_Loss%.4f-Val_Loss%.4f-AP_50_%.4f.pth'%((epoch+1),train_loss,val_loss,AP_50))
                    torch.save(model.state_dict(), save_model_dir + 'FB_object_detect_model.pth')
            
            if largest_AP_50 > AP_50:
                # To load the largest_AP_50 model
                net = load_model(model, save_model_dir + largest_AP_50_model_name, cuda=Cuda)
                net = net.train()

            if (epoch+1)>=2:
                draw_curve_loss(epoch+1, train_loss.item(), val_loss.item(), log_pic_name_loss)
            if (epoch+1)>=30:
                draw_curve_ap50(epoch+1, AP_50, log_pic_name_ap50)
            lr_scheduler.step()