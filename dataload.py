import torch
import os
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import av
    
class MyDataset(Dataset):
    def __init__(self, json_name ,tokenizer=None):

        self.data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

        self.json_path = os.path.join(self.data_root, json_name)

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
    
    def get_txt_feats(self,idx):
        '''
        return: torch.tensor
        '''
        txt_feat_pth =os.path.join(self.data_root, 'txt_feat','q_subtitle_{}.pt'.format(idx))
        txt_feat = torch.load(txt_feat_pth).to('cpu')
        return txt_feat

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
        res['txt_feat'] = self.get_txt_feats(id)

        # 计算视频的时长
        video_path = os.path.join(self.data_root,'videos', video_name+'.mp4')
        container = av.open(video_path)
        seg_len=container.streams.video[0].frames 
        frame_rate = container.streams.video[0].average_rate 
        duration = seg_len/frame_rate # 视频秒数
        res['duration'] = duration

        
        txt_list = [question]
        token_types=[]

        for item in res['subtitles']:
            txt_list.append(item['subtitle'])
        SEP_token = ' ' + self.tokenizer.sep_token + ' '
        txt = SEP_token.join(txt_list)

        txt_ids = self.data_tokenize(txt)


        for token_id in txt_ids:
            if token_id == self.tokenizer.sep_token_id:
                token_types.append(1)
            else:
                token_types.append(0)
        res['q_srt_ids'],l ,res['q_srt_mask']  = self.get_padded_tensor(txt_ids, max_lens = 512)

        # 将token_type短的补全到512，长的截断
        if len(token_types) > 512:
            token_types = token_types[:512]
        else:
            token_types = token_types + [0] * (512 - l)

        res['token_type']= torch.LongTensor(token_types)

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
    txt_features = []
    token_types = []
    video_masks = []
    txt_masks = []
    ious = []
    for batch in batches:
        id = batch['id']
        video_feats = batch['video_feats']
        question = batch['question']
        start_second = batch['start_second']
        end_second = batch['end_second']
        txt_feat = batch['txt_feat']
        token_type = batch['token_type']
        txt_mask = batch['q_srt_mask']

        # 生成768*768的方阵
        iou = np.zeros((768, 768))

        # 判断duration是否超过768，如果超过，就将start_second和end_second都除以duration//768+1
        duration = batch['duration']
        if duration > 768:
            start_second = start_second // (duration//768+1)
            end_second = end_second // (duration//768+1)
        
        # 在[start_second, end_second]的位置为1，其他位置为0
        iou[start_second][end_second] = 1

        ious.append(torch.FloatTensor(iou))
        # 计算video_mask，其中非0的部分为1，0的部分为0
        video_mask = video_feats.sum(-1) != 0
        vid_features.append(torch.FloatTensor(video_feats))
        txt_masks.append(txt_mask)
        txt_features.append(txt_feat)
        token_types.append(token_type)
        video_masks.append(torch.LongTensor(video_mask))
    vid_feat_res = torch.stack(vid_features, dim=0)
    txt_feat_res = torch.stack(txt_features, dim=0)
    token_type_res = torch.stack(token_types, dim=0)
    video_mask_res = torch.stack(video_masks, dim=0)
    txt_mask_res = torch.stack(txt_masks, dim=0)
    return vid_feat_res, txt_feat_res,txt_mask_res, token_type_res, video_mask_res, ious
    

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
