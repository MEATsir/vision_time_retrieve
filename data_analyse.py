import av
import numpy as np
from tqdm import tqdm
import os
import json


# 读取文件夹中所有的视频，返回所有视频的时间，帧数，帧率，并输出最高的时间和帧数
def video_analyse(video_dir):
    video_list = os.listdir(video_dir)
    duration_lst = []
    frames_lst = []
    frames_rate_lst = []
    for video in tqdm(video_list):
        video_path = os.path.join(video_dir, video)
        
        container = av.open(video_path)
        seg_len=container.streams.video[0].frames 
        frame_rate = container.streams.video[0].average_rate 
        duration = seg_len/frame_rate # 视频秒数

        duration_lst.append(duration)
        frames_lst.append(seg_len)
        frames_rate_lst.append(frame_rate)
    
    print('最高帧率为:',max(frames_rate_lst))
    print('最高时间为:',max(duration_lst),'秒')
    print('最高帧数为:',max(frames_lst))
    print('时长列表',duration_lst)
    print('帧数',frames_lst)
    print('帧率',frames_rate_lst)

# 读取文件夹里的字幕，返回字幕的长度和
def subtitle_analyse(subtitle_dir):
    subtitle_list = os.listdir(subtitle_dir)
    len_list  = []
    for subtitle_name in subtitle_list:
        subtitle_path = os.path.join(subtitle_dir, subtitle_name)
        with open(subtitle_path, 'r') as f:
            subtitles = f.read().splitlines()
        sub_batch = []
            
        for string in range(len(subtitles)):
            # 判断字符串是否是纯数字
            if subtitles[string].isdigit():
                if not subtitles[string+1]:
                    continue
                sub_batch.append(subtitles[string+2])
        len_list.append(len(sub_batch))
    print(max(len_list))


    max_500_cout = 0
    for lens in len_list:
        if lens >=400:
            max_500_cout+= 1
    print('总长度超过400的字幕数量为',max_500_cout)

    min_5_cout = 0
    for lens in len_list:
        if lens <=10:
            min_5_cout += 1
    print('总长度小于10的字幕数量为',min_5_cout)


def feat_analyse(feat_dir,video_dir):
    feat_list = os.listdir(feat_dir)
    cout = 0
    for video_name in tqdm(os.listdir(video_dir)):
        if video_name.split('.')[0] + '.npy' not in feat_list:
            print(video_name)
#获取当前程序路径
path = os.path.dirname(os.path.abspath(__file__))

fusion_feat_dir = 'data/fusion_vid_feat/'
feat_dir = 'data/raw_vid_feat/'
video_dir = 'data/videos/'
subtitle_dir = 'data/subtitles'

fusion_feat_path = os.path.join(path, fusion_feat_dir)
feat_path = os.path.join(path, feat_dir)
video_path = os.path.join(path, video_dir)
subtitle_path = os.path.join(path, subtitle_dir)

feat_analyse(fusion_feat_path,video_path)
        
