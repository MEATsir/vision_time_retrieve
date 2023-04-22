# vision_time_retrieve

# NLPCC TASK5 中医视频问答时刻检测

## 1、数据处理

将数据中的字幕串联起来，利用sep标记进行分割，将处理得到的字幕串和问题分别通过预训练模型（bigbird中文）得到特征。

## 2、tgt设计

我们将时刻预测分为了start时刻预测与end时刻预测，我们将所有的视频长度缩放到768以便模型能够处理。对于时刻预测，我们采用的是分类方法，将任务看做两组768的分类。然而768种类太多，数据量过少，因此采用了层次化分类以便模型更好的预测。

## 3、模型

attention + conv1 + relu + linear * 4 得到四个概率。 
