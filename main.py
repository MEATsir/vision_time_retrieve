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

def train_model(model,optimizer,scaler,scheduler,train_loader):  # 训练一个epoch
    model.train()

    losses = AverageMeter()
    optimizer.zero_grad()

    tk = tqdm(train_loader, total=len(train_loader), position=0, leave=True)
    for step, (vid_feat, txt_feat, txt_mask,token_type, video_mask, ious) in enumerate(tk):
        with autocast():
            output = model(txt_feat,txt_mask,token_types = token_type,
                          ious=ious,vfeats=vid_feat,vfeats_mask=video_mask)


        loss = output['CEloss']
        scaler.scale(loss).backward()

        if ((step + 1) % CFG['accum_iter'] == 0) or ((step + 1) == len(train_loader)):  # 梯度累加
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
        losses.update(loss.item())
        tk.set_postfix(loss=losses.avg)
        if step == 0:
            log(['Start Train:','Now epoch:{}'.format(epoch),'Now Loss：{}'.format(str(loss.item())),'all of the step:{}'.format(len(tk))],path)

    log(['Now Loss：{}'.format(str(loss.item())),'Avg Loss：{}'.format(losses.avg),'End this round of training'],path)
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

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", default='base', type=str)
    parser.add_argument("--seed", default=2, type=int)
    parser.add_argument("--maxlen", default=1300, type=int)
    parser.add_argument("--epochs", default=32, type=int)
    parser.add_argument("--batchsize", default=1, type=int)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--weight_decay", default=0, type=float)
    parser.add_argument("--device", default=1, type=float)
    parser.add_argument("--json_name",default='CMIVQA_Train_Dev.json' ,type=str)
    parser.add_argument('--tokenizer',default="hfl/chinese-roberta-wwm-ext",type=str)
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
    }

    accelerator = Accelerator()
    seed_everything(CFG['seed'])
    torch.cuda.set_device(CFG['device'])
    device = accelerator.device

    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)

    train_data = MyDataset(args.json_name,tokenizer=tokenizer)
    train_loader = DataLoader(train_data,args.batchsize,True,collate_fn=collect_fn)
    best_acc = 0


    from model import GlobalSpanModel
    model = GlobalSpanModel()
    model.to(device=device)

    optimizer = AdamW(model.parameters(), weight_decay=CFG['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    scheduler = get_cosine_schedule_with_warmup(optimizer, len(train_loader) // CFG['accum_iter'],
                                                CFG['epochs'] * len(train_loader) // CFG['accum_iter'])
    
    train_loader = accelerator.prepare(train_loader)
    scaler = GradScaler()
    path = log_start()
    log(get_args(args),path)

    
    for epoch in range(CFG['epochs']):
        train_model(model,optimizer,scaler,scheduler,train_loader)
