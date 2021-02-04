import argparse
import csv
import os
import time

import torch
import torch.utils.data
import numpy as np
import tqdm

import yolo_model
import utils.datasets
import utils.utils

def evaluate(model, path, iou_thres, nms_thres, image_size, batch_size, num_workers):
    #모델을 evaluation mode로 설정
    model.eval()

    dataset = utils.datasets.ListDataset(path, image_size, augement=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=num_workers,
                                             collate_fn=dataset.collate_fn)

    labels=[]
    sample_metrics = []
    entire_time = 0
    for _, images, targets in tqdm.tqdm(dataloader, desc='evaluate method', leave=False):
        if targets is None:


