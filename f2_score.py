import numpy as np
from typing import List
import torch
from torchvision.ops import box_iou

def calculate_score(preds: List[torch.Tensor], gts: List[torch.Tensor], iou_th: float):
    num_tp = 0
    num_fp = 0
    num_fn = 0
    for p, GT in zip(preds, gts):
        if len(p) and len(GT):
            gt = GT.clone()
            gt[:,2] = gt[:,0] + gt[:,2]
            gt[:,3] = gt[:,1] + gt[:,3]
            pp = p.clone()
            pp[:,2] = pp[:,0] + pp[:,2]
            pp[:,3] = pp[:,1] + pp[:,3]
            iou_matrix = box_iou(pp, gt)
            tp = len(torch.where(iou_matrix.max(0)[0] >= iou_th)[0])
            fp = len(p) - tp
            fn = len(torch.where(iou_matrix.max(0)[0] < iou_th)[0])
            num_tp += tp
            num_fp += fp
            num_fn += fn
        elif len(p)==0 and len(GT):
            num_fn += len(GT)
        elif len(p) and len(GT)==0:
            num_fp += len(p)
    score = 5*num_tp/(5*num_tp + 4*num_fn + num_fp)
    return score