import cv2
import math
import numpy as np
import os
import torch
import torch.nn as nn
import transformers
import torch.utils.checkpoint
import av

from tqdm import tqdm
from tqdm import trange
from transformers import AutoImageProcessor, VideoMAEModel

# 1、处理视频特征

# 读取pyav文件，将文件处理为np文件
def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

# 返回切片
def sample_frame_indices(start_idx, end_idx , clip_len):
    
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

def pad_video_seq(sequences, max_length=1024):
    if max_length is None:
        max_length = max([vfeat.shape[0] for vfeat in sequences])
    feature_length = sequences[0].shape[1]
    sequence_padded, sequence_length = [], []
    for seq in sequences:
        add_length = max_length - seq.shape[0]
        sequence_length.append(seq.shape[0])
        if add_length > 0:
            add_feature = np.zeros(shape=[add_length, feature_length], dtype=np.float32)
            seq_ = np.concatenate([seq, add_feature], axis=0)
        else:
            seq_ = seq
        sequence_padded.append(seq_)
    return sequence_padded, sequence_length

# 读取视频，返回以视频名字命名的特征
def video2feat(video_path,image_processor,model, clip_len = 16):
    container = av.open(video_path)
    
    seg_len=container.streams.video[0].frames # 180000+
    frame_rate = container.streams.video[0].average_rate # 30 or 25
    duration = seg_len/frame_rate # 视频秒数 有可能不是整数 # 6000s
    # rate/16 * [1,2,3,4...duration]

    avg_16_frame = frame_rate/clip_len

    for i in trange(int(duration)):
        # sample 16 frames
        indices = np.array([int(j*avg_16_frame+i*frame_rate)+1 for j in range(clip_len)]).astype(np.int64)
        video = read_video_pyav(container, indices)
        # prepare video for the model
        inputs = image_processor(list(video), return_tensors="pt")
        # forward pass
        outputs = model(**inputs.to(model.device))
        last_hidden_states = outputs.last_hidden_state.to('cpu')

        if i == 0:
            feats = last_hidden_states
        else:
            feats = torch.cat((feats,last_hidden_states),0)
    return feats #[秒数，1568, 768]


# 读取文件夹中所有的视频，返回所有视频的特征，并将其按照视频的名字保存为np文件到指定文件夹
def video2feat_dir(video_dir,feat_dir, image_processor, model, clip_len = 16):
    video_list = os.listdir(video_dir)[:120]
    for video in tqdm(video_list):
        video_path = os.path.join(video_dir, video)
        feats = video2feat(video_path, image_processor, model, clip_len = clip_len)
        np.save(os.path.join(feat_dir, video.split('.')[0]), feats)


def vision_feat_fusion(vision_feat):
    # 按照特征第二个纬度平均
    vision_feat = vision_feat.mean(1)

    # 判断vision_feat第一个纬度是否超过768
    if vision_feat.shape[0] > 768:
        decline_rate = vision_feat.shape[0] // 768 + 1

        # 按照decline_rate进行降采样
        # vision_feat = vision_feat[::decline_rate]

        # 按照decline_rate的步长在第一维度进行平均
        # vision_feat = vision_feat.reshape(-1, decline_rate, vision_feat.shape[1]).mean(1) 
        new_visual_feature = []

        for i in range(vision_feat.shape[0] // decline_rate ):
            s_idx = i * decline_rate
            e_idx = min((i + 1) * decline_rate, vision_feat.shape[0])
            new_visual_feature.append(np.mean(vision_feat[s_idx:e_idx], axis=0))
        vision_feat = np.asarray(new_visual_feature)
    # 在第一维度进行0填充
    vision_feat = np.pad(vision_feat, ((0, 768 - vision_feat.shape[0]), (0, 0)), 'constant')
    return vision_feat

def video_pre_process():
    # 当前路径
    cur_path = os.path.dirname(os.path.abspath(__file__))

    
    feat_path = os.path.join(cur_path, 'data/raw_vid_feat/')
    # 判断是否有路路径/raw_vid_feat/文件夹，如果没有则提取初步特征
    if not os.path.exists(feat_path):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        video_path = os.path.join(cur_path, 'data/videos/')

        # 如果没有feat_path文件夹，则创建
        if not os.path.exists(feat_path):
            os.mkdir(feat_path)

        # load model
        model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
        model.to(device)
        model.eval()
        for pra in model.parameters():
            pra.requires_grad = False
        image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
        video2feat_dir(video_path, feat_path, image_processor, model, clip_len = 16)
    
    # 读取特征文件夹中的所有特征文件
    feat_list = os.listdir(feat_path)

    # 创建特征融合文件夹
    fusion_feat_path = os.path.join(cur_path, 'data/fusion_vid_feat/')
    if not os.path.exists(fusion_feat_path):
        os.mkdir(fusion_feat_path)

    # 遍历feat_list中的所有文件
    for feat_name in tqdm(feat_list):
        # 读取文件
        feat = np.load(os.path.join(feat_path, feat_name))
        # 将文件进行特征融合
        feat = vision_feat_fusion(feat)
        # 保存文件
        np.save(os.path.join(fusion_feat_path, feat_name), feat)

# 2、处理文本特征

import pandas as pd
from transformers import BertTokenizer, BertModel

def get_data(json_path):
    # 利用pandas读取json文件
    json_data = pd.read_json(json_path)
    return json_data
    
def data_tokenize(txt,tokenizer):
    '''
    利用编码器返回编码列表

    return:
    [0,2384,12345,12235,3556,6778,3]
    '''
    unpaded = tokenizer.tokenize(txt)
    txt_ids = tokenizer.convert_tokens_to_ids(unpaded)
    return txt_ids
    

def get_padded_tensor(txt,tokenizer, max_lens = 0):

    length = len(txt)
    if length > max_lens:
        out = txt[:max_lens ]
        length = max_lens 
    else:
        out = [tokenizer.pad_token_id] * max_lens 
        out[:length] = txt
    out = torch.LongTensor(out)
    mask = out != 0
    return out , length, mask

def get_subtitles(data_root, video_name):
    '''
    return: list
    '''
    subtitle_pth =os.path.join(data_root, video_name.split('.')[0])
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

def q_srt_preprocess(max_lens = 512,device = 'cuda'):
    # 当前路径
    cur_path = os.path.dirname(os.path.abspath(__file__))
    subtitle_pth = os.path.join(cur_path, 'data/subtitles/')

    # 读取json文件
    json_path = os.path.join(cur_path, 'data/CMIVQA_Train_Dev.json')
    json_data = get_data(json_path)

    # 读取预训练模型,并冻结参数
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    model = BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext")
    for para in model.parameters():
        para.requires_grad = False
    model.to(device)
    model.eval()

    # 创建txt_feat文件夹
    txt_path = os.path.join(cur_path, 'data/q_srts_feat/')
    if not os.path.exists(txt_path):
        os.mkdir(txt_path)
    

    for idx in tqdm(range(len(json_data))):
        data = []
        for key in json_data.keys():
            data.append(json_data[key][idx])
        id , video_name, question, start_second, end_second = data
        subtitles = get_subtitles(subtitle_pth, video_name)    
        txt_list = [question]

        for item in subtitles:
            txt_list.append(item['subtitle'])
        SEP_token = ' ' + tokenizer.sep_token + ' '
        txt = SEP_token.join(txt_list)
        txt_ids = data_tokenize(txt,tokenizer=tokenizer)
        txt_ids, length, mask = get_padded_tensor(txt_ids, tokenizer, max_lens)

        input_ids = txt_ids.unsqueeze(0).to(device)
        mask = mask.unsqueeze(0).to(device)

        output = model(input_ids, attention_mask = mask)
        save = output.last_hidden_state.squeeze(0)

        save_name ='q_subtitle_' + str(id) + '.pt'
        torch.save(save , os.path.join(txt_path, save_name))
    

def padding_q_and_srt(max_lens = 512,device = 'cuda'):
    cur_path = os.path.dirname(os.path.abspath(__file__))
    subtitle_pth = os.path.join(cur_path, 'data/subtitles/')

    # 读取json文件
    json_path = os.path.join(cur_path, 'data/CMIVQA_Train_Dev.json')
    json_data = get_data(json_path)

    # 读取预训练模型,并冻结参数
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    model = BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext")
    for para in model.parameters():
        para.requires_grad = False
    model.to(device)
    model.eval()

    # 创建subtitles_feat文件夹
    subtitles_feat_path = os.path.join(cur_path, 'data/pad_subtitles_feat/')
    if not os.path.exists(subtitles_feat_path):
        os.mkdir(subtitles_feat_path)  
    
    # 创建q_feat文件夹
    q_feat_path = os.path.join(cur_path, 'data/pad_q_feat/')
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
        txt_ids, length, mask = get_padded_tensor(txt_ids, tokenizer, max_lens)

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
        id , video_name, question, start_second, end_second = data

        q_ids = data_tokenize(question,tokenizer=tokenizer)
        q_ids, length, mask = get_padded_tensor(q_ids, tokenizer, 52)

        input_ids = q_ids.unsqueeze(0).to(device)
        mask = mask.unsqueeze(0).to(device)

        output = model(input_ids, attention_mask = mask)
        save = output.last_hidden_state.squeeze(0)

        save_name ='pad_question_' + str(id) + '.pt'
        torch.save(save , os.path.join(q_feat_path, save_name))


if __name__ == '__main__':
    padding_q_and_srt()
