import torch


def bbox_wh_iou(wh, gwh):
    # anchor box의 위치 상관없이 gt box와 예측한 anchor가 같은 중심이라고 할때, 크기가 가장 좋은 anchor 종류 고르기
    # wh : 예측 anchor box의 wh -> [2]
    # gwh : target box의 wh -> [2]
    gwh = gwh.t()
    w, h = wh[0], wh[1]
    gw, gh = gwh[0], gwh[1]
    inter_area = torch.min(w, gw) * torch.min(h, gh)
    union_area = (w * h + 1e-16) + gw * gh - inter_area
    return inter_area/union_area


def bbox_iou(box1, box2, x1y1x2y2=True):
    # 위치 포함하여 예측한 box와 target box의 iou
    # box1 -> [n, 4]
    # box2 -> [n, 4]
    if not x1y1x2y2:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2]/2, box1[:, 0] + box1[:, 2]/2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    #intersection 사각형 좌표
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    #intersection area
    # clamp는 min 이상인 것만 남기기
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    #union area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area + 1e-16)
    return iou


def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):
    # pred_boxes -> [batch_size, anchor, gird, grid, 4] -> 4 = 예측 box의 x, y, w, h
    # pred_cls -> [batch, anchor, grid, grid, num_class]
    # targets : ground truth -> [batch_size, 6] -> 6 = num(gt box의 num, 배치 인덱스), class, x, y, w, h
    # anchors -> [3, 2] -> 3= 개수, 2 = w, h

    nB = pred_boxes.size(0) # 배치 사이즈 1
    nA = pred_boxes.size(1) # anchor 개수 3
    nC = pred_cls.size(-1) #클래스 수 80
    nG = pred_boxes.size(2) #grid 사이즈

    #output
    obj_mask = torch.zeros(nB, nA, nG, nG, dtype=torch.bool) #물체 있는 경우 1
    noobj_mask = torch.ones(nB, nA, nG, nG, dtype=torch.bool) #물체 있는 경우 0
    class_mask = torch.zeros(nB, nA, nG, nG, dtype=torch.float) #클래스를 맞춘 경우
    iou_mask = torch.zeros(nB, nA, nG, nG, dtype=torch.float) # iou가 얼마인지
    tx = torch.zeros(nB, nA, nG, nG, dtype=torch.float) #target의 x 값(0~1)
    ty = torch.zeros(nB, nA, nG, nG, dtype=torch.float) #target의 y 값(0~1)
    tw = torch.zeros(nB, nA, nG, nG, dtype=torch.float) #target의 w 값(0~1)
    th = torch.zeros(nB, nA, nG, nG, dtype=torch.float) #target의 h 값(0~1)
    tcls = torch.zeros(nB, nA, nG, nG, nC, dtype=torch.float) #정답 클래스

    #target bow의 위치
    target_boxes = target[:, 2:6]*nG #(0~1)범위에서 (0~13)범위, 즉 grid 크기로 변경
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]

    #최고의 iou를 갖는 anchor box 구하기
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors]) #[3 , n] -> 3 = anchor
    _, best_ious_idx = ious.max(0) #하나의 열에서 최대인것

    #target 값 분리
    b, target_label = target[:,:2].long().t()
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t() #정수로 만들어주기 위해

    #mask 설정
    obj_mask[b, best_ious_idx, gj, gi] = 1 #target box의 위치에서 적합한 anchor box 부분에 1 -> 물체가 있는 곳 1
    noobj_mask[b, best_ious_idx, gj, gi] = 0 #물체가 있는 곳은 0으로 설정

    #ious 가 treshold보다 크면 noobj_mask를 0으로 설정
    #ious = [3, n] -> [n,3] = 어떤 target box 인지, 어떤 anchor box인지
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious> ignore_thres, gj[i], gi[i]] = 0 # 물체가 있을 수 있다고 판단하여 0으로 설정

    #target box 좌표 설정(0~1사이의 값으로)
    tx[b, best_ious_idx, gj, gi] = gx - gx.floor() #floor()은 내림(ex> 1.5 -> 1)
    ty[b, best_ious_idx, gj, gi] = gy - gy.floor()

    #너비, 높이 설정
    tw[b, best_ious_idx, gj, gi] = torch.log(gw / anchors[best_ious_idx][:, 0] + 1e-16)
    th[b, best_ious_idx, gj, gi] = torch.log(gh / anchors[best_ious_idx][:, 1] + 1e-16)

    #class label값
    tcls[b, best_ious_idx, gj, gi, target_label] = 1

    #label을 정확하게 예측했는지, 최적의 andhor의 iou는 무엇인지(이때는 위치 포함)
    class_mask[b, best_ious_idx, gj, gi] = (pred_cls[b, best_ious_idx, gj, gi].argmax(-1) == target_label).float()
    iou_mask[b, best_ious_idx, gj, gi] = bbox_iou(pred_boxes[b, best_ious_idx, gj, gi], target_boxes, x1y1x2y2 = False)

    tconf = obj_mask.float()
    return iou_mask, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf


def xywh2xyxy(x):
    y = x.new(x.shape)
    y[...,0] = x[...,0] - x[...,2]/2
    y[...,1] = x[...,1] - x[...,3]/2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def non_max_suppression(prediction, conf_th, nms_th):
    prediction[...,:4] = xywh2xyxy(prediction[...,:4])
    output = [None for _ in range(len(prediction))] # 1길이 list

    for image_i, image_pred in enumerate(prediction): # 순서번호, 예측값 #image_i는 배치, image_pred.shape = (10647,85)
        image_pred = image_pred[image_pred[:,4]>=conf_th]  # (n, 85) -> conf가 th보다 큰 anchor만 남김
        if not image_pred.size(0):
            continue

        score = image_pred[:,4] * image_pred[:,5:].max(1)[0] #(n) -> 85중에 가장 큰 값만(인덱스 말고)

        image_pred = image_pred[(-score).argsort()] #내림차순 #(n, 85) #score가 큰 순서대로 anchor 정렬
        class_confs, class_preds = image_pred[:,5:].max(1, keepdim=True) #80중에 가장 큰 class의 값과 인덱스 반환 #각각 (n,1)
        detections = torch.cat((image_pred[:,:5], class_confs.float(), class_preds.floar()),1) #(n, 7)

        keep_boxes = []
        while detections.size(0):
            # 가장 큰 스코어를 가진 박스와 iou 비교하여 th보다 크면 겹친다고 판단
            large_overlap = bbox_iou(detections[0,:4].unsqueeze(0), detections[:,:4]) > nms_th #(4) -> (1,4)와 (n, 4)비교 #결과 [1,0,1,1,0]
            label_match = detections[0,-1] == detections[:,-1] # 예측한 class가 같으면 match #결과[1, 0,0,0,0]
            invalid = large_overlap & label_match # iou가 많이 겹치고 그 클래스가 같으면 1, 아니면 0 #결과 [1, 0,0,0,0]
            weights = detections[invalid, 4] # invalid한 anchor의 objectness score
            detections[0,:4] = (weights * detections[invalid, :4]).sum(0)/weights.sum() #invalid한 box끼리 계산하여 하나로 만듦
            keep_boxes += [detections[0]]
            detections = detections[~invalid] #나머지 남은 박스들끼리 또 비교
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output #(batch, pred_boxes_num, 7) # 7-> x,y,w,h,conf,class_conf,class_pred


def parse_data_config(path:str):
    # 데이터셋 설정 파일을 parse(분석)함
    options = {}
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip() #앞뒤 whitespace(띄어쓰기, 탭, 엔터)제거
        key, value = line.split('=')
        options[key.strip()] = value.strip()

    return options


def load_classes(path:str):
    # 클래스 이름 불러오기
    with open(path, 'r') as f:
        names = f.readlines()
    for i, name in enumerate(names):
        names[i] = name.strip()
    return names

def init_weight_normal(m):
    # 정규분포 형태로 가중치 초기화
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.kaiming_normal_(m.weight.data, 0.1)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)