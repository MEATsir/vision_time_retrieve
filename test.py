from pre_process import *

import torch
import os
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import av
import json
from Crossatt import *

def test_padding_q_and_srt(device = 'cuda:5'):
    cur_path = os.path.dirname(os.path.abspath(__file__))
    subtitle_pth = os.path.join(cur_path, 'testA/subtitles/')

    # 读取json文件
    json_path = os.path.join(cur_path, 'testA/dataset_testA_for_track1.json')
    json_data = get_data(json_path)

    # 读取预训练模型,并冻结参数
    tokenizer = JiebaTokenizer.from_pretrained('Lowin/chinese-bigbird-base-4096')
    model = BigBirdModel.from_pretrained('Lowin/chinese-bigbird-base-4096')
    for para in model.parameters():
        para.requires_grad = False
    model.to(device)
    model.eval()

    # 创建subtitles_feat文件夹
    subtitles_feat_path = os.path.join(cur_path, 'testA/pad_subtitles_feat/')
    if not os.path.exists(subtitles_feat_path):
        os.mkdir(subtitles_feat_path)  
    
    # 创建q_feat文件夹
    q_feat_path = os.path.join(cur_path, 'testA/pad_q_feat/')
    if not os.path.exists(q_feat_path):
        os.mkdir(q_feat_path)
    
    # 遍历videos文件夹，获取视频名称
    videos = os.listdir(subtitle_pth)
    videos = [video.split('.')[0]+'.mp4' for video in videos]

    # 按照视频名称读取字幕文件
    for video in tqdm(videos):
        subtitles = get_subtitles(subtitle_pth, video)
        txt_list = []
        for item in subtitles:
            txt_list.append(item['subtitle'])
        SEP_token = ' ' + tokenizer.sep_token + ' '
        txt = SEP_token.join(txt_list)
        txt_ids = data_tokenize(txt,tokenizer=tokenizer)
        txt_ids, length, mask = get_padded_tensor(txt_ids, tokenizer, max_lens = 4096)

        input_ids = txt_ids.unsqueeze(0).to(device)
        mask = mask.unsqueeze(0).to(device)

        output = model(input_ids, attention_mask = mask)
        save = output.last_hidden_state.squeeze(0)

        save_name ='pad_subtitle_' + video.split('.')[0] + '.pt'
        torch.save(save , os.path.join(subtitles_feat_path, save_name))

    
    for idx in tqdm(range(len(json_data))):
        data = []
        for key in json_data.keys():
            data.append(json_data[key][idx])
        id , video_name, question = data

        q_ids = data_tokenize(question,tokenizer=tokenizer)
        q_ids, length, mask = get_padded_tensor(q_ids, tokenizer, 52)

        input_ids = q_ids.unsqueeze(0).to(device)
        mask = mask.unsqueeze(0).to(device)

        output = model(input_ids, attention_mask = mask)
        save = output.last_hidden_state.squeeze(0)

        save_name ='pad_question_' + str(id) + '.pt'
        torch.save(save , os.path.join(q_feat_path, save_name))


class TestDataset(Dataset):
    def __init__(self, json_name ,tokenizer=None):

        self.data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'testA')
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
        id , video_name, question = data


        res ={}
        res['id'] = id
        res['video_name'] = video_name
        res['question'] = question
        res['subtitles'] = self.get_subtitles(video_name)
        res['q_feat'],res['sub_feat'] = self.get_q_and_sub_feats(id,video_name)

        # 计算视频的时长
        video_path = os.path.join(self.data_root,'videos', video_name+'.mp4')
        container = av.open(video_path)
        seg_len=container.streams.video[0].frames 
        frame_rate = container.streams.video[0].average_rate 
        duration = seg_len/frame_rate # 视频秒数
        res['duration'] = duration

        # 获取问题和字幕的mask
        sub_list = []
        for item in res['subtitles']:
            sub_list.append(item['subtitle'])
        SEP_token = ' ' + self.tokenizer.sep_token + ' '
        subs = SEP_token.join(sub_list)
        sub_ids = self.data_tokenize(subs)
        sub_ids, length, sub_mask = self.get_padded_tensor(sub_ids, max_lens = 4096)

        res['sub_msk'] = sub_mask
        res['subtitles_ids'] = sub_ids
        q_ids = self.data_tokenize(question)
        q_ids, length, q_mask = self.get_padded_tensor(q_ids, max_lens = 52)
        res['q_msk'] = q_mask

        return res

def test_collect_fn(batches):
    '''
    batch: dict

    '''
    ids = []
    q_features = []
    sub_features = []
    sub_idss = []
    q_masks = []
    sub_masks = []

    for batch in batches:
        id = batch['id']
        sub_ids = batch['subtitles_ids']

        q_feat = batch['q_feat']
        sub_feat = batch['sub_feat']
        q_mask = batch['q_msk']
        sub_mask = batch['sub_msk']


        

        # 判断duration是否超过768，如果超过，就将start_second和end_second都除以duration//768+1
        duration = batch['duration']
        

        # 计算video_mask，其中非0的部分为1，0的部分为0
        ids.append(id)
        q_features.append(q_feat)
        sub_features.append(sub_feat)
        q_masks.append(q_mask)
        sub_masks.append(sub_mask)
        sub_idss.append(sub_ids)


    sub_ids_res = torch.stack(sub_idss, dim=0)
    q_feat_res = torch.stack(q_features, dim=0)
    sub_feat_res = torch.stack(sub_features, dim=0)
    q_mask_res = torch.stack(q_masks, dim=0)
    sub_mask_res = torch.stack(sub_masks, dim=0)

    return q_feat_res,q_mask_res, sub_ids_res,sub_feat_res, sub_mask_res,duration,ids
    


def test_batches(model,test_loader,device):
    model.eval()
    res_= []
    with torch.no_grad():
        for q_feat_res,q_mask_res, sub_ids_res,sub_feat_res, sub_mask_res,duration,id in tqdm(test_loader):
            q_feat_res = q_feat_res.to(device)
            q_mask_res = q_mask_res.to(device)
            sub_ids_res = sub_ids_res.to(device)
            sub_feat_res = sub_feat_res.to(device)
            sub_mask_res = sub_mask_res.to(device)
            start1_logits, start2_logits, end1_logits, end2_logits = model(q_feat_res,None,None,sub_feat_res,None)
            
            # 层次化分类目标还原
            start1_p = (torch.max(start1_logits,dim =1)[1].to('cpu').numpy()+1)%32
            start2_p = (torch.max(start2_logits,dim =1)[1].to('cpu').numpy()+1) % 24 
            end1_p = (torch.max(end1_logits,dim =1)[1].to('cpu').numpy()+1) %32
            end2_p = (torch.max(end2_logits,dim =1)[1].to('cpu').numpy()+1) % 24

            start_pred = 24 * start1_p + start2_p
            end_pred = 24 * end1_p + end2_p

            # 将start_pred和end_pred作比较，如果start_pred>end_pred，就将end_pred = start_pred
            for i in range(len(start_pred)):
                start_pred,end_pred = end_pred,start_pred

            res_.append([id,start_pred,end_pred])

    # 将id，start_pred,end_pred写入json文件
    reslst = []
    for item in res_:
        res = {}
        id = item[0][0]
        start_pred = item[1][0]
        end_pred = item[2][0]
        res['id'] = int(id)
        res['start_second'] = int(start_pred)
        res['end_second'] = int(end_pred)
        reslst.append(res)
    
    with open('/home/bma/CMIVQA/testA/result.json','w') as f:
        json.dump(reslst,f,indent=1)

    return res_


if __name__ == '__main__':
    # 载入模型


    from torch.utils.data import DataLoader

    tokenizer = JiebaTokenizer.from_pretrained("Lowin/chinese-bigbird-base-4096")
    model = SpanExtraction()
    model.load_state_dict(torch.load('/home/bma/CMIVQA/checkpoint/model_226.pth'))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # 载入数据
    test_dataset = TestDataset('/home/bma/CMIVQA/testA/dataset_testA_for_track1.json',tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=test_collect_fn)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    res = test_batches(model,test_loader,device)
