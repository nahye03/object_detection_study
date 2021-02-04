import torch
import torch.nn as nn
import utils


def dbl(in_ch, out_ch, kernel_size, stride=1, padding=0):
    'conv+bn+leaky relu로 구성된 DBL 블럭'
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.1)
    )


class ResUnit(nn.Module):
    def __init__(self, in_ch):
        super(ResUnit, self).__init__()
        middle_ch = int(in_ch / 2)

        self.conv1 = dbl(in_ch, middle_ch, 1)
        self.conv2 = dbl(middle_ch, in_ch, 3, padding=1)

    def forward(self, inputs):
        res = inputs
        x = self.conv1(inputs)
        x = self.conv2(x)

        return res+x


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, n_iter):
        super(ResBlock, self).__init__()

        self.conv1 = dbl(in_ch, out_ch, 3, 2, 1)
        self.res_units = []
        for i in range(n_iter):
            self.res_units.append(ResUnit(out_ch))

    def forward(self, inputs):
        x = self.conv1(inputs)
        for res_unit in self.res_units:
            x = res_unit(x)
        return x


class darknet53(nn.Module):
    def __init__(self):
        super(darknet53, self).__init__()

        self.conv1 = dbl(3, 32, 3, padding=1)
        self.layer1 = ResBlock(32, 64, 1)
        self.layer2 = ResBlock(64, 128, 2)
        self.layer3 = ResBlock(128, 256, 8)
        self.layer4 = ResBlock(256, 512, 8)
        self.layer5 = ResBlock(512, 1024, 4)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        out1 = self.layer3(x)
        out2 = self.layer4(out1)
        out3 = self.layer5(out2)

        return out1, out2, out3


def dblset(in_ch, out_ch):
    return nn.Sequential(
        dbl(in_ch, out_ch, 1, 1),
        dbl(out_ch, in_ch, 3, 1, 1),
        dbl(in_ch, out_ch, 1, 1),
        dbl(out_ch, in_ch, 3, 1, 1),
        dbl(in_ch, out_ch, 1, 1)
    )


def final_block(in_ch, out_ch):
    mid_ch = 2 * in_ch
    return nn.Sequential(
        dbl(in_ch, mid_ch, 3, 1, 1),
        nn.Conv2d(mid_ch, out_ch, 1)
    )


class YOLODetection(nn.Module):
    def __init__(self, anchors, img_size, num_classes):
        super(YOLODetection, self).__init__()

        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.img_size = img_size

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

        self.ignore_th = 0.5
        self.obj_scale = 1
        self.no_obj_scale = 100
        self.metrics ={}

    def forward(self, x, targets):
        # x : feature map -> [batch_size, final_ch, grid, grid]
        # targets : ground truth -> [batch_size, 6] -> 6 = num(gt box의 num, 배치 인덱스), class, x, y, w, h

        num_batches = x.size(0)
        grid_size = x.size(2)

        #출력값 형태 변환
        prediction = (
            x.view(num_batches, self.num_anchors, self.num_classes+5, grid_size, grid_size)
                .permute(0, 1, 3, 4, 2).contiguous()) #메모리 연속 할달

        #get outputs
        #format : [batch, anchors, grid, grid]
        cx = torch.sigmoid(prediction[...,0]) #예측 box의 중심 x좌표
        cy = torch.sigmoid(prediction[...,1]) #예측 box의 중심 y좌표
        w = prediction[...,2] #예측 box의 w
        h = prediction[...,3] #예측 box의 h
        pred_conf = torch.sigmoid(prediction[...,4]) #confidence
        pred_cls = torch.sigmoid(prediction[...,5:]) #class 확률

        #offset 구하기
        stride = self.img_size / grid_size
        ''' grid_x=([[0],[1],[2],[3],...,[16]],
                    [[0],[1],[2],[3],...,[16]],
                            ...
                    [[0],[1],[2],[3],...,[16]])
            grid_y=([[0],[0],[0],...,[0]],
                    [[1],[1],[1],...,[1]],
                            ...
                    [[16],[16],[16],...,[16]])'''
        grid_x = torch.arange(grid_size, dtype=torch.float).repeat(grid_size, 1).view([1,1,grid_size, grid_size])
        grid_y = torch.arange(grid_size, dtype=torch.float).repeat(grid_size, 1).t().view([1, 1, grid_size, grid_size])
        scaled_anchor = torch.as_tensor([(a_w/stride, a_h/stride) for a_w, a_h in self.anchors],
                                        dtype=torch.float)
        anchor_w = scaled_anchor[:,0].view((1, self.num_anchors, 1, 1))
        anchor_h = scaled_anchor[:,1].view((1, self.num_anchors, 1, 1))

        #예측한 anchor 좌표 구하기
        pred_boxes = torch.zeros_like(prediction[...,:4]) #x, y, w, h
        pred_boxes[...,0] = cx + grid_x
        pred_boxes[...,1] = cy + grid_y
        pred_boxes[...,2] = torch.exp(w)*anchor_w
        pred_boxes[...,3] = torch.exp(h)*anchor_h

        pred = (pred_boxes.view(num_batches, -1 , 4)*stride, #(1, 3*grid*grid, 4)
                pred_conf.view(num_batches, -1, 1), #(1, 3*grid*grid, 1)
                pred_cls.view(num_batches, -1, self.num_classes)) #(1, 3*grid*grid, 80)

        output = torch.cat(pred, -1) #(1, 3*grid*grid, 85)

        if targets is None:
            return output, 0

        iou_scores, class_mask, obj_mask, no_obj_mask, tx, ty, tw, th, tcls, tconf = utils.build_targets(
            pred_boxes = pred_boxes,
            pred_cls = pred_cls,
            target = targets,
            anchors = scaled_anchor,
            ignore_thres = self.ignore_th
        )

        #loss 구하기
        loss_x = self.mse_loss(cx[obj_mask], tx[obj_mask])
        loss_y = self.mse_loss(cy[obj_mask], ty[obj_mask])
        loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
        loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
        loss_bbox = loss_x + loss_y + loss_w + loss_h

        loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
        loss_conf_no_obj = self.bce_loss(pred_conf[no_obj_mask], tconf[no_obj_mask])
        loss_conf = self.obj_scale * loss_conf_obj + self.no_obj_scale * loss_conf_no_obj #패널티를 주기위해

        loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
        loss_layer = loss_bbox + loss_conf + loss_cls

        # Metrics
        conf50 = (pred_conf > 0.5).float()
        iou50 = (iou_scores > 0.5).float()
        iou75 = (iou_scores > 0.75).float()
        detected_mask = conf50 * class_mask * tconf
        cls_acc = 100 * class_mask[obj_mask].mean()
        conf_obj = pred_conf[obj_mask].mean()
        conf_no_obj = pred_conf[no_obj_mask].mean()
        precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
        recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
        recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

        # Write loss and metrics
        self.metrics = {
            "loss_x": loss_x.detach().cpu().item(),
            "loss_y": loss_y.detach().cpu().item(),
            "loss_w": loss_w.detach().cpu().item(),
            "loss_h": loss_h.detach().cpu().item(),
            "loss_bbox": loss_bbox.detach().cpu().item(),
            "loss_conf": loss_conf.detach().cpu().item(),
            "loss_cls": loss_cls.detach().cpu().item(),
            "loss_layer": loss_layer.detach().cpu().item(),
            "cls_acc": cls_acc.detach().cpu().item(),
            "conf_obj": conf_obj.detach().cpu().item(),
            "conf_no_obj": conf_no_obj.detach().cpu().item(),
            "precision": precision.detach().cpu().item(),
            "recall50": recall50.detach().cpu().item(),
            "recall75": recall75.detach().cpu().item()
        }


        return output, loss_layer


class YOLOv3(nn.Module):
    def __init__(self, img_size: int, num_classes: int):
        super(YOLOv3, self).__init__()

        anchors = {'scale1': [(10, 13), (16, 30), (33, 23)],
                   'scale2': [(30, 61), (62, 45), (59, 119)],
                   'scale3': [(116, 90), (156, 198), (373, 326)]}
        final_out_ch = 3*(4+1+num_classes)

        self.darknet = darknet53()

        self.out3_dblset = dblset(1024, 512)
        self.out3_final = final_block(512, final_out_ch)
        self.yolo_layer3 = YOLODetection(anchors['scale3'], img_size, num_classes)


        self.out3_branch = dbl(512, 256, 1, 1)
        self.out3_up = nn.Upsample(scale_factor=2, mode='nearest')
        self.out2_dblset = dblset(768, 256)
        self.out2_final = final_block(256, final_out_ch)
        self.yolo_layer2 = YOLODetection(anchors['scale2'], img_size, num_classes)


        self.out2_branch = dbl(256, 128, 1, 1)
        self.out2_up = nn.Upsample(scale_factor=2, mode='nearest')
        self.out1_dblset = dblset(384, 128)
        self.out1_final = final_block(128, final_out_ch)
        self.yolo_layer1 = YOLODetection(anchors['scale1'], img_size, num_classes)

        self.yolo_layers = [self.yolo_layer1, self.yolo_layer2, self.yolo_layer3]

    def forward(self, img, targets=None):
        loss = 0

        outs = self.darknet(img)

        #13x13 feature map
        x3 = self.out3_dblset(outs[2])
        fea3 = self.out3_final(x3)
        yolo_output3, layer_loss = self.yolo_layer3(fea3, targets)
        loss += layer_loss

        #26x26 feature map
        x3 = self.out3_branch(x3)
        x3 = self.out3_up(x3)
        x2 = torch.cat([x3, outs[1]], 1)
        x2 = self.out2_dblset(x2)
        fea2 = self.out2_final(x2)
        yolo_output2, layer_loss = self.yolo_layer2(fea2, targets)
        loss += layer_loss

        #52x52 feature map
        x2 = self.out2_branch(x2)
        x2 = self.out2_up(x2)
        x1 = torch.cat([x2, outs[0]], 1)
        x1 = self.out1_dblset(x1)
        fea1 = self.out1_final(x1)
        yolo_output1, layer_loss = self.yolo_layer1(fea1, targets)
        loss += layer_loss

        yolo_outputs = [yolo_output1, yolo_output2, yolo_output3] #(1, 3*grid*grid, 85)가 3개
        yolo_outputs = torch.cat(yolo_outputs, 1).detach() #(1, 3*grid*grid *3, 85) #detach()는 autograd 추적 중단

        return yolo_outputs if targets is None else (loss, yolo_outputs)



def main():
    in_tensor = torch.rand(1, 3, 416, 416)
    target = torch.rand(1, 6)
    model = YOLOv3(img_size=416, num_classes=20)
    loss, yolo_outputs = model(in_tensor, target)
    print(yolo_outputs.shape)
    print(loss)


if __name__ =="__main__":
    main()

