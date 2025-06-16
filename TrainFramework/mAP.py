import torch
from collections import Counter



def IOU(box1, box2):
    """
        计算IOU
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    inter_rect_x1 = max(b1_x1, b2_x1)
    inter_rect_y1 = max(b1_y1, b2_y1)
    inter_rect_x2 = min(b1_x2, b2_x2)
    inter_rect_y2 = min(b1_y2, b2_y2)

    inter_area = max(inter_rect_x2 - inter_rect_x1 + 1, 0) * \
                 max(inter_rect_y2 - inter_rect_y1 + 1, 0)
                 
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def mean_average_precision(pred_oriented_objs,true_oriented_objs,iou_threshold):

    epsilon=1e-6#防止分母为0
    detections = pred_oriented_objs
    ground_truths = true_oriented_objs
    #img 0 has 3 bboxes
    #img 1 has 5 bboxes
    #就像这样：amount_bboxes={0:3,1:5}
    #统计每一张图片中真实框的个数,train_idx指示了图片的编号以区分每张图片
    amount_bboxes=Counter(gt.image_id for gt in ground_truths)

    for key,val in amount_bboxes.items():
        amount_bboxes[key]=torch.zeros(val)#置0，表示这些真实框初始时都没有与任何预测框匹配
    #此时，amount_bboxes={0:torch.tensor([0,0,0]),1:torch.tensor([0,0,0,0,0])}

    #将预测框按照置信度从大到小排序
    detections.sort(key=lambda x:x.score,reverse=True)

    #初始化TP,FP
    TP=torch.zeros(len(detections))
    FP=torch.zeros(len(detections))

        #TP+FN就是当前类别GT框的总数，是固定的
    total_true_bboxes=len(ground_truths)
    
    #如果一个GT框都没有，那么直接返回
    if total_true_bboxes == 0:
        return 0, 0, 0

    #对于每个预测框，先找到它所在图片中的所有真实框，然后计算预测框与每一个真实框之间的IoU，大于IoU阈值且该真实框没有与其他预测框匹配，则置该预测框的预测结果为TP，否则为FP
    for detection_idx,detection in enumerate(detections):
        #在计算IoU时，只能是同一张图片内的框做，不同图片之间不能做
        #图片的编号存在第0个维度
        #于是下面这句代码的作用是：找到当前预测框detection所在图片中的所有真实框，用于计算IoU
        ground_truth_img=[bbox for bbox in ground_truths if bbox.image_id==detection.image_id]

        best_iou=0
        for idx,gt in enumerate(ground_truth_img):
            #计算当前预测框detection与它所在图片内的每一个真实框的IoU
            iou=IOU(detection.bbox,gt.bbox)
            if iou >best_iou:
                best_iou=iou
                best_gt_idx=idx
        if best_iou>iou_threshold:
            #这里的detection[0]是amount_bboxes的一个key，表示图片的编号，best_gt_idx是该key对应的value中真实框的下标
            if amount_bboxes[detection.image_id][best_gt_idx]==0:#只有没被占用的真实框才能用，0表示未被占用（占用：该真实框与某预测框匹配【两者IoU大于设定的IoU阈值】）
                TP[detection_idx]=1#该预测框为TP
                amount_bboxes[detection.image_id][best_gt_idx]=1#将该真实框标记为已经用过了，不能再用于其他预测框。因为一个预测框最多只能对应一个真实框（最多：IoU小于IoU阈值时，预测框没有对应的真实框)
            else:
                FP[detection_idx]=1#虽然该预测框与真实框中的一个框之间的IoU大于IoU阈值，但是这个真实框已经与其他预测框匹配，因此该预测框为FP
        else:
            FP[detection_idx]=1#该预测框与真实框中的每一个框之间的IoU都小于IoU阈值，因此该预测框直接为FP
    TP_cumsum=torch.cumsum(TP,dim=0)
    FP_cumsum=torch.cumsum(FP,dim=0)

    TP_sum = torch.sum(TP)
    FP_sum = torch.sum(FP)
    
    #套公式
    recalls=TP_cumsum/(total_true_bboxes+epsilon)
    precisions=torch.divide(TP_cumsum,(TP_cumsum+FP_cumsum+epsilon))

    recalls_value=TP_sum/(total_true_bboxes+epsilon)
    precisions_value=TP_sum/(TP_sum+FP_sum+epsilon)

    #把[0,1]这个点加入其中
    precisions=torch.cat((torch.tensor([1]),precisions))
    recalls=torch.cat((torch.tensor([0]),recalls))
    #使用trapz计算AP
    average_precision_value = torch.trapz(precisions,recalls)

    return average_precision_value, recalls_value, precisions_value