import argparse
import os
import time

import torch
import torch.utils.data
import torch.utils.tensorboard # 학습 시각화
import tqdm

import yolo_model
# import utils.datasets
import utils.utils
import utils.datasets

# from test import evaluate

# 호출 당시 인자값을 줘서 동작을 다르게 하고 싶은 경우 -> argparse 모듈 사용
parser = argparse.ArgumentParser() # 인자값 받을 수 있는 인스턴스 생성
parser.add_argument("--epoch", type=int, default=100, help='num of epoch')
parser.add_argument("--gradient_accumulation", type=int, default=1, help='num od gradient accums before step') #gradient 축적
parser.add_argument('--multiscale_training', type=bool, default=True, help='allow for multi-scale training')
parser.add_argument('--batch_size', type=int, default=32, help='size of each image batch')
parser.add_argument('--num_workers', type=int, default=8, help='num of cpu threads to use during batch generation')
parser.add_argument('--data_config', type=str, default='config/voc.data', help='path of data config file')
parser.add_argument('--pretrained_weights', type=str, default='weights/darknet53.conv.74',help="if specified starts from checkpoint model")
parser.add_argument('--image_size', type=int, default=416, help='size of image dimension')
args = parser.parse_args() # 입력받은 인자값을 args에 저장
print(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
now = time.strftime('%y%m%d_%H%M%S', time.localtime(time.time()))

# tensorboard writer 설정
log_dir = os.path.join('logs', now)
os.makedirs(log_dir, exist_ok=True) # exist_ok=True : 폴더가 존재하지 않으면 생성하고, 존재하는 경우에는 아무것도 하지 않음
writer = torch.utils.tensorboard.SummaryWriter(log_dir)

# 데이터셋 설정값 가져오기
data_config = utils.utils.parse_data_config(args.data_config)
train_path = data_config['train']
valid_path = data_config['valid']
num_classes = int(data_config['classes'])
class_name = utils.utils.load_classes(data_config['names'])

# 모델 준비
model = yolo_model.YOLOv3(args.image_size, num_classes)
model.apply(utils.utils.init_weight_normal)
# if args.pretrained_weights.endswith('.pth'):
#     model.load_state_dict(torch.load(args.pretrained_weights))
# else:
#     model.load_darknet_weights(args.pretrained_weights)

# 데이터셋, 데이터로더 설정
dataset = utils.datasets.ListDataset(train_path, args.image_size, augment = True, multiscale = args.multiscale_training)
# dataloader -> 반복자
dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=args.batch_size,
                                         shuffle=True,
                                         num_workers=args.num_workers,
                                         pin_memory=True,
                                         collate_fn=dataset.collate_fn)
#optimizer 설정
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#lr scheduler 설정
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10,gamma=0.5)

#현재 배치 손실값을 출력하는 tqdm 설정/ 작업진행률
loss_log = tqdm.tqdm(total=0, position=2, bar_format='{desc}', leave=False)

#train
for epoch in tqdm.tqdm(range(args.epoch), desc='Epoch'): #desc : 작업에 대한 설명
    #모델을 train모드로 설정
    model.train()

    for batch_idx,(_, images, targets) in enumerate(tqdm.tqdm(dataloader, desc='Batch',leave='False')):
        step = len(dataloader) * epoch + batch_idx

        loss, output = model(images, targets) #순전파
        loss.backward() # 역전파

        if step % args.gradient_accumulation == 0:
            optimizer.step() #매개변수 갱신
            optimizer.zero_grad() # 갱신할 변수 들어 대한 모든 변화도를 0으로

        loss_log.set_description_str('Loss: {:.6f}'.format(loss.item()))

        tensorboard_log = []
        for i, yolo_layer in enumerate(model.yolo_layers):
            writer.add_scalar('loss_bbox_{}'.format(i + 1), yolo_layer.metrics['loss_bbox'], step)
            writer.add_scalar('loss_conf_{}'.format(i + 1), yolo_layer.metrics['loss_conf'], step)
            writer.add_scalar('loss_cls_{}'.format(i + 1), yolo_layer.metrics['loss_cls'], step)
            writer.add_scalar('loss_layer_{}'.format(i + 1), yolo_layer.metrics['loss_layer'], step)
        writer.add_scalar('total_loss', loss.item(), step)

    scheduler.step()

    #검증 데이터셋으로 모델 평가
    precision, recall, AP, f1, _, _, _ = evaluate(model,
                                                  path =valid_path,
                                                  iou_thres=0.5,
                                                  conf_thres = 0.5,   )




