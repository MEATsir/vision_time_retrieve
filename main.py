import json
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm,trange
import random
import os
import time
from transformers import BertTokenizer,AutoModelForSequenceClassification,AutoTokenizer,AdamW,get_cosine_schedule_with_warmup,AutoModelForSequenceClassification
from accelerate import Accelerator
import glob
from dataload import *

def get_args(args):
    l = []
    for k in list(vars(args).keys()):
        l.append(('%s: %s' % (k, vars(args)[k])))
    return l


def log(text,path):
    with open(path+'/log.txt','a',encoding='utf-8') as f:
        f.write('-----------------{}-----------------'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        for i in text:
            f.write(i)
            print(i)
            f.write('\n')
        f.write('\n')

class AverageMeter:  # 为了tqdm实时显示loss和acc
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def log_start():
    # 获取当前程序路径
    path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(path, 'logs')
    if not os.path.exists(path):
        os.mkdir(path)
    # 根据当前时间戳创建log文件夹
    log_path = os.path.join(path, '#{}log'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    return log_path


def calculate_batch_iou(start_bs,end_bs, start_t, end_t):#s1<s2,start end是金标
    iou = 0
    for i in range(len(start_bs)):
        second1 = start_bs[i]
        second2 = end_bs[i]
        start = start_t[i]
        end = end_t[i]
        union = min(second2, end) - max(second1, start)
        inter = max(second2, end) - min(second1, start)
        iou += 1.0 * max(union / inter,0)
    return max(0.0, iou/len(start_bs))


def train_model(model,optimizer,criterion ,scaler,scheduler,train_loader,save_pth):  # 训练一个epoch
    model.train()

    losses = AverageMeter()
    optimizer.zero_grad()

    tk = tqdm(train_loader, total=len(train_loader), position=0, leave=True)

    max_iou = 0
    iou_sum = 0
    for step, (vid_feat, q_feat,q_mask,sub_ids, sub_feat, sub_mask, video_mask, ious, strat1_tgt ,start2_tgt,end1_tgt,end2_tgt,test) in enumerate(tk):
        with autocast():
            output = model(q_feat,q_mask,sub_ids,sub_feat,sub_mask,
                          ious=ious,vfeats=vid_feat,vfeats_mask=video_mask,start1_tgt=strat1_tgt,start2_tgt = start2_tgt,end1_tgt=end1_tgt,end2_tgt=end2_tgt)


        start1_logits,start2_logits,end1_logits,end2_logits = output
        loss = 0.7*(criterion(start1_logits,strat1_tgt) + criterion(end1_logits,end1_tgt))+0.3*(criterion(start2_logits,start2_tgt)+criterion(end2_logits,end2_tgt))


        # 层次化分类目标还原
        start1_p = (torch.max(start1_logits,dim =1)[1].to('cpu').numpy()+1)%32
        start2_p = (torch.max(start2_logits,dim =1)[1].to('cpu').numpy()+1) % 24 
        end1_p = (torch.max(end1_logits,dim =1)[1].to('cpu').numpy()+1) %32
        end2_p = (torch.max(end2_logits,dim =1)[1].to('cpu').numpy()+1) % 24

        start1_t = (torch.max(strat1_tgt, dim=1)[1].to('cpu').numpy() +1) % 32
        start2_t = (torch.max(start2_tgt, dim=1)[1].to('cpu').numpy()+1)%24
        end1_t = (torch.max(end1_tgt, dim=1)[1].to('cpu').numpy()+1) %32
        end2_t = (torch.max(end2_tgt, dim=1)[1].to('cpu').numpy()+1)%24
        

        start_pred = 24 * start1_p + start2_p
        end_pred = 24 * end1_p + end2_p
        start_true = 24* start1_t + start2_t
        end_true = 24 *  end1_t + end2_t
        iou = calculate_batch_iou(start_pred,end_pred,start_true,end_true)
        iou_sum += iou

        # 还原结束

        scaler.scale(loss).backward()

        if ((step + 1) % CFG['accum_iter'] == 0) or ((step + 1) == len(train_loader)):  # 梯度累加
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
        losses.update(loss.item())
        tk.set_postfix(loss=loss.item())

    # 如果miou大于之前的最大值，保存模型
    if iou_sum/len(train_loader) > max_iou:
        max_iou = iou_sum/len(train_loader)
        torch.save(model.state_dict(), os.path.join(save_pth, 'model_{}.pth'.format(epoch)))
    log(['Start Train:','Now epoch:{}'.format(epoch),'Now Loss：{}'.format(str(loss.item())),'Avg Loss：{}'.format(losses.avg),'miou:{}'.format(iou_sum/len(train_loader)),'End this round of training'],path)
    
    
    return losses.avg

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    from pre_process import JiebaTokenizer

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", default='base', type=str)
    parser.add_argument("--seed", default=2023, type=int)
    parser.add_argument("--maxlen", default=1300, type=int)
    parser.add_argument("--epochs", default=2000, type=int)

    parser.add_argument("--batchsize", default=3, type=int)

    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--weight_decay", default=0.00001, type=float)
    parser.add_argument("--device", default=6, type=float)
    parser.add_argument("--json_name",default='CMIVQA_Train_Dev.json' ,type=str)
    parser.add_argument('--tokenizer',default="Lowin/chinese-bigbird-base-4096",type=str)
    args = parser.parse_args()
    CFG = {
        'seed': args.seed,
        'max_len': args.maxlen,
        'epochs': args.epochs,
        'train_bs': 1,
        'valid_bs': 1,
        'lr': args.lr,
        'num_workers': args.num_workers,
        'accum_iter': args.batchsize,
        'weight_decay': args.weight_decay,
        'device': args.device,
        'save_pth': '/home/bma/CMIVQA/checkpoint/',
    }

    accelerator = Accelerator()
    seed_everything(CFG['seed'])
    torch.cuda.set_device(CFG['device'])
    device = accelerator.device

    tokenizer = JiebaTokenizer.from_pretrained(args.tokenizer)

    train_data = MyDataset(args.json_name,tokenizer=tokenizer)
    train_loader = DataLoader(train_data,args.batchsize,True,collate_fn=collect_fn)
    best_acc = 0


    from Crossatt import SpanExtraction
    model = SpanExtraction()
    model.to(device=device)

    optimizer = AdamW(model.parameters(),lr = CFG['lr'], weight_decay=CFG['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    scheduler = get_cosine_schedule_with_warmup(optimizer, len(train_loader) // CFG['accum_iter'],
                                                CFG['epochs'] * len(train_loader) // CFG['accum_iter'])
    
    train_loader = accelerator.prepare(train_loader)
    scaler = GradScaler()
    path = log_start()
    log(get_args(args),path)

    save_pth = CFG['save_pth']
    for epoch in range(CFG['epochs']):
        train_model(model,optimizer,criterion,scaler,scheduler,train_loader,save_pth=save_pth)
