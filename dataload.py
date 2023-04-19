import torch
import os
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import av
import pickle    

class MyDataset(Dataset):
    def __init__(self, json_name ,tokenizer=None):

        self.data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

        self.json_path = os.path.join(self.data_root, json_name)
        self.duration_dict = pickle.load(open(os.path.join(self.data_root, 'duration_dict.pkl'), 'rb'))

        self.df = self.get_data()
        self.tokenizer = tokenizer

    def get_data(self):
        # 利用pandas读取json文件
        json_data = pd.read_json(self.json_path)
        return json_data


    def __len__(self):

        return len(self.df)
    
    def data_tokenize(self,txt):
        '''
        利用编码器返回编码列表

        return:
        [0,2384,12345,12235,3556,6778,3]
        '''
        unpaded = self.tokenizer.tokenize(txt)
        txt_ids = self.tokenizer.convert_tokens_to_ids(unpaded)
        return txt_ids
    

    def get_padded_tensor(self, txt, max_lens = 0):

        length = len(txt)
        if length > max_lens:
            out = txt[:max_lens ]
            length = max_lens 
        else:
            out = [self.tokenizer.pad_token_id] * max_lens 
            out[:length] = txt
        out = torch.LongTensor(out)
        mask = out != 0
        return out , length, mask
    
    def get_video_feats(self,video_name):
        '''
        return: torch.tensor
        '''
        feat_pth =os.path.join(self.data_root,'fusion_vid_feat', video_name.split('.')[0] + '.npy')

        video_feats = np.load(feat_pth)
        return video_feats
    
    def get_q_sub_feats(self,idx):
        '''
        return: torch.tensor
        '''
        txt_feat_pth =os.path.join(self.data_root, 'txt_feat','q_subtitle_{}.pt'.format(idx))
        txt_feat = torch.load(txt_feat_pth).to('cpu')
        return txt_feat
    
    def get_q_and_sub_feats(self,idx,video_name):
        '''
        return: torch.tensor
        '''
        q_feat_pth =os.path.join(self.data_root, 'pad_q_feat','pad_question_{}.pt'.format(idx))
        q_feat = torch.load(q_feat_pth).to('cpu')

        sub_feat_pth =os.path.join(self.data_root, 'pad_subtitles_feat','pad_subtitle_'+video_name + '.pt')
        sub_feat = torch.load(sub_feat_pth).to('cpu')

        return q_feat,sub_feat

    def get_subtitles(self,video_name):
        '''
        return: list
        '''
        subtitle_pth =os.path.join(self.data_root, 'subtitles',video_name.split('.')[0])
        subtitle_pth += '.srt'
        with open(subtitle_pth,'r') as f:
            subtitles = f.read().splitlines()
        res = []

        for string in range(len(subtitles)):
            # 判断字符串是否是纯数字
            if subtitles[string].isdigit():
                if not subtitles[string+1]:
                    continue
                mid_res  = {}
                mid_res['subtitle'] = subtitles[string+2]

                time_str = subtitles[string+1].split('-->') # ['00:00:00,000' '00:00:00,000']
                start_time = time_str[0].split(':') # [hour,minute,second]
                end_time = time_str[1].split(':')

                mid_res['start_second'] =  float(start_time[0])*3600 + float(start_time[1])*60 + float(start_time[2].split(',')[0])
                mid_res['end_second'] = float(end_time[0])*3600 + float(end_time[1])*60 + float(end_time[2].split(',')[0])
                res.append(mid_res) 
        return res

    def __getitem__(self, idx):
        '''
        return: dict
                -> {'id': id -> int,
                    'video_name': video_name,
                    'video_feats': video_feats -> ndarray,
                    'question': question -> str,
                    'start_second': start_second -> float,
                    'end_second': end_second -> float,
                    'subtitles': subtitles -> list[dict,dict,dict...],

        '''
        
        data = []
        for key in self.df.keys():
            data.append(self.df[key][idx])
        id , video_name, question, start_second, end_second = data


        res ={}
        res['id'] = id
        res['video_name'] = video_name
        res['video_feats'] = self.get_video_feats(video_name)
        res['question'] = question
        res['start_second'] = start_second
        res['end_second'] = end_second
        res['subtitles'] = self.get_subtitles(video_name)
        res['q_feat'],res['sub_feat'] = self.get_q_and_sub_feats(id,video_name)

        # 计算视频的时长

        res['duration'] = self.duration_dict[video_name]

        # 获取问题和字幕的mask
        sub_list = []
        for item in res['subtitles']:
            sub_list.append(item['subtitle'])
        SEP_token = ' ' + self.tokenizer.sep_token + ' '
        subs = SEP_token.join(sub_list)
        sub_ids = self.data_tokenize(subs)
        sub_ids, length, sub_mask = self.get_padded_tensor(sub_ids, max_lens = 512)
        res['sub_msk'] = sub_mask

        q_ids = self.data_tokenize(question)
        q_ids, length, q_mask = self.get_padded_tensor(q_ids, max_lens = 52)
        res['q_msk'] = q_mask

        return res

def calculate_iou(second1, second2, start, end):#s1<s2,start end是金标
    union = min(second2, end) - max(second1, start)
    inter = max(second2, end) - min(second1, start)
    iou = 1.0 * union / inter
    return max(0.0, iou)

def calculate_iou_accuracy(ious, threshold):#ious是预测的sample的iou，threshold是阈值0.3 0.5 0.7
    total_size = float(len(ious))
    count = 0
    for iou in ious:
        if iou >= threshold:
            count += 1
    return float(count) / total_size * 100.0

def calculate_miou(ious):#计算mIOU
    return np.mean(ious)



def collect_fn(batches):
    '''
    batch: dict

    '''
    vid_features = []
    q_features = []
    sub_features = []
    q_masks = []
    sub_masks = []
    token_types = []
    video_masks = []
    txt_masks = []
    start_seconds = []
    end_seconds = []
    ious = []
    for batch in batches:
        id = batch['id']
        video_feats = batch['video_feats']
        question = batch['question']
        start_second = batch['start_second']
        end_second = batch['end_second']
        q_feat = batch['q_feat']
        sub_feat = batch['sub_feat']
        q_mask = batch['q_msk']
        sub_mask = batch['sub_msk']

        # 生成768*768的方阵
        iou = np.zeros((768, 768))

        # 判断duration是否超过768，如果超过，就将start_second和end_second都除以duration//768+1
        duration = batch['duration']
        if duration > 768:
            start_second = start_second // (duration//768+1)
            end_second = end_second // (duration//768+1)

        # 生成长度为768的向量，其中start_second的位置为1，其他位置为0
        start_ls = [0] * 768
        start_ls[start_second] = 1
        # 生成长度为768的向量，其中end_second的位置为1，其他位置为0
        end_ls = [0] * 768
        end_ls[end_second] = 1
        
        # 在[start_second, end_second]的位置为1，其他位置为0
        iou[start_second][end_second] = 1

        ious.append(torch.FloatTensor(iou))
        start_seconds.append(torch.LongTensor(start_ls))
        end_seconds.append(torch.LongTensor(end_ls))
        # 计算video_mask，其中非0的部分为1，0的部分为0
        video_mask = video_feats.sum(-1) != 0
        vid_features.append(torch.FloatTensor(video_feats))
        q_features.append(q_feat)
        sub_features.append(sub_feat)
        q_masks.append(q_mask)
        sub_masks.append(sub_mask)
        video_masks.append(torch.FloatTensor(video_mask))
    vid_feat_res = torch.stack(vid_features, dim=0)
    q_feat_res = torch.stack(q_features, dim=0)
    sub_feat_res = torch.stack(sub_features, dim=0)
    q_mask_res = torch.stack(q_masks, dim=0)
    sub_mask_res = torch.stack(sub_masks, dim=0)
    video_mask_res = torch.stack(video_masks, dim=0)
    start_seconds_res = torch.stack(start_seconds, dim=0)
    end_seconds_res = torch.stack(end_seconds, dim=0)

    return vid_feat_res, q_feat_res,q_mask_res, sub_feat_res, sub_mask_res, video_mask_res, ious, start_seconds_res, end_seconds_res
    

if __name__ == "__main__":
    from transformers import BertTokenizer
    from torch.utils.data import DataLoader

    json_name = 'CMIVQA_Train_Dev.json'
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

    data = MyDataset(json_name=json_name,tokenizer=tokenizer)
    dataloader = DataLoader(data, batch_size=2, shuffle=True, num_workers=0, collate_fn=collect_fn)
    for i, batch in enumerate(dataloader):
        for item in batch:
            print(item.shape)