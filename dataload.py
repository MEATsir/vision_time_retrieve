import torch
import os
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

    
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
        
        # txt_list = [question]

        # for item in res['subtitles']:
        #     txt_list.append(item['subtitle'])
        # SEP_token = ' ' + self.tokenizer.sep_token + ' '
        # txt = SEP_token.join(txt_list)
        # txt_ids = self.data_tokenize(txt)
        # res['q_srt_ids'] = torch.LongTensor(txt_ids)

        return res

def collect_fn(batch):
    '''
    batch: dict

    '''
    id = batch['id']
    video_name = batch['video_name']
    video_feats = batch['video_feats']
    question = batch['question']
    start_second = batch['start_second']
    end_second = batch['end_second']
    subtitles = batch['subtitles']
    txt_feat = batch['txt_feat']

    return id, video_name, video_feats, question, start_second, end_second, subtitles, txt_feat


if __name__ == "__main__":
    from transformers import BertTokenizer

    json_name = 'CMIVQA_Train_Dev.json'
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

    data = MyDataset(json_name=json_name,tokenizer=tokenizer)
    print(data[0])