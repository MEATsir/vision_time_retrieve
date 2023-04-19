import torch
import torch.nn as nn



def mask_logits(inputs, mask, mask_value=-1e30):
    mask = mask.type(torch.float32) 
    return inputs + (1.0 - mask) * mask_value



class CQAttention(nn.Module):
    def __init__(self, dim, drop_rate=0.0):
        super(CQAttention, self).__init__()
        w4C = torch.empty(dim, 1)
        w4Q = torch.empty(dim, 1)
        w4mlu = torch.empty(1, 1, dim)
        nn.init.xavier_uniform_(w4C)
        nn.init.xavier_uniform_(w4Q)
        nn.init.xavier_uniform_(w4mlu)
        self.w4C = nn.Parameter(w4C, requires_grad=True)
        self.w4Q = nn.Parameter(w4Q, requires_grad=True)
        self.w4mlu = nn.Parameter(w4mlu, requires_grad=True)
        self.dropout = nn.Dropout(p=drop_rate)
        self.cqa_linear = Conv1D(in_dim=4 * dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, context, query, c_mask, q_mask):
        score = self.trilinear_attention(context, query)  # (batch_size, c_seq_len, q_seq_len)context:2,768,768 query:2,307,768 score:2,768,307
        score_ = nn.Softmax(dim=2)(mask_logits(score, q_mask.unsqueeze(1)))  # (batch_size, c_seq_len, q_seq_len)
        score_t = nn.Softmax(dim=1)(mask_logits(score, c_mask.unsqueeze(2)))  # (batch_size, c_seq_len, q_seq_len)
        score_t = score_t.transpose(1, 2)  # (batch_size, q_seq_len, c_seq_len) 2,307,768
        c2q = torch.matmul(score_, query)  # (batch_size, c_seq_len, dim) 2,768,768
        q2c = torch.matmul(torch.matmul(score_, score_t), context)  # (batch_size, c_seq_len, dim) 2,768,768
        output = torch.cat([context, c2q, torch.mul(context, c2q), torch.mul(context, q2c)], dim=2) # (batch_size, c_seq_len, 4 * dim) 2,768,3072
        output = self.cqa_linear(output)  # (batch_size, c_seq_len, dim) #2,768,768
        return output

    def trilinear_attention(self, context, query):
        batch_size, c_seq_len, dim = context.shape
        batch_size, q_seq_len, dim = query.shape
        context = self.dropout(context)
        query = self.dropout(query)
        subres0 = torch.matmul(context, self.w4C).expand([-1, -1, q_seq_len])  # (batch_size, c_seq_len, q_seq_len)
        subres1 = torch.matmul(query, self.w4Q).transpose(1, 2).expand([-1, c_seq_len, -1])
        subres2 = torch.matmul(context * self.w4mlu, query.transpose(1, 2))
        res = subres0 + subres1 + subres2  # (batch_size, c_seq_len, q_seq_len)
        return res
    


class WeightedPool(nn.Module):
    def __init__(self, dim):
        super(WeightedPool, self).__init__()
        weight = torch.empty(dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def forward(self, x, mask):
        alpha = torch.tensordot(x, self.weight, dims=1)  # shape = (batch_size, seq_length, 1) 2,307,1
        alpha = mask_logits(alpha, mask=mask.unsqueeze(2))
        alphas = nn.Softmax(dim=1)(alpha)
        pooled_x = torch.matmul(x.transpose(1, 2), alphas)  # (batch_size, dim, 1) 2,768,1
        pooled_x = pooled_x.squeeze(2) #2,768
        return pooled_x
    

class Conv1D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=True):
        super(Conv1D, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size, padding=padding,
                                stride=stride, bias=bias)

    def forward(self, x):
        # suppose all the input with shape (batch_size, seq_len, dim)
        x = x.transpose(1, 2)  # (batch_size, dim, seq_len)
        x = self.conv1d(x)
        return x.transpose(1, 2)  # (batch_size, seq_len, dim)
    

class CQConcatenate(nn.Module):
    def __init__(self, dim):
        super(CQConcatenate, self).__init__()
        self.weighted_pool = WeightedPool(dim=dim)
        self.conv1d = Conv1D(in_dim=2 * dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, context, query, q_mask):
        pooled_query = self.weighted_pool(query, q_mask)#query:2,307,768 q_mask:2,307 pooled_query:2,768
        _, c_seq_len, _ = context.shape #2,768,768
        pooled_query = pooled_query.unsqueeze(1).repeat(1, c_seq_len, 1)  # (batch_size, c_seq_len, dim)
        output = torch.cat([context, pooled_query], dim=2)  # (batch_size, c_seq_len, 2*dim) 2,768,1536
        output = self.conv1d(output) #2,768,768
        return output
    
class HighLightLayer(nn.Module):
    def __init__(self, dim):
        super(HighLightLayer, self).__init__()
        self.conv1d = Conv1D(in_dim=dim, out_dim=1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x, mask):
        # compute logits
        logits = self.conv1d(x) #x:2,768,768 logits:2,768,1
        logits = logits.squeeze(2) #2,768
        logits = mask_logits(logits, mask)
        # compute score
        scores = nn.Sigmoid()(logits) #2,768
        return scores

    @staticmethod
    def compute_loss(scores, labels, mask, epsilon=1e-12):
        labels = labels.type(torch.float32)
        weights = torch.where(labels == 0.0, labels + 1.0, 2.0 * labels)
        loss_per_location = nn.BCELoss(reduction='none')(scores, labels)
        loss_per_location = loss_per_location * weights
        mask = mask.type(torch.float32)
        loss = torch.sum(loss_per_location * mask) / (torch.sum(mask) + epsilon)
        return loss




class CrossAttention(nn.Module):
    def __init__(self, video_dim, txt_dim, hidden_dim):
        super(CrossAttention, self).__init__()
        self.video_dim = video_dim
        self.txt_dim = txt_dim
        self.hidden_dim = hidden_dim
        self.video_fc = nn.Linear(video_dim, hidden_dim)
        self.txt_fc = nn.Linear(txt_dim, hidden_dim)
        self.cross_att = nn.MultiheadAttention(hidden_dim, num_heads=8)
        self.fc = nn.Linear(hidden_dim*2, 1)
        self.Dropout = nn.Dropout(0.1)

    def forward(self, video_feat, text_feat, video_padding_mask):
        # video_feat: (batch_size, seq_len, video_dim)
        # text_feat: (batch_size, txt_seq_len, txt_dim)
        # video_fc_out: (batch_size, seq_len, hidden_dim)
        # txt_fc_out: (batch_size, txt_seq_len, hidden_dim)
        video_fc_out = self.video_fc(video_feat)
        video_fc_out = self.Dropout(video_fc_out)
        txt_fc_out = self.txt_fc(text_feat)
        txt_fc_out = self.Dropout(txt_fc_out)
        # cross_att_out: (batch_size, seq_len, hidden_dim)
        cross_att_out, _ = self.cross_att(txt_fc_out.transpose(0, 1), 
                                          video_fc_out.transpose(0, 1), 
                                          video_fc_out.transpose(0, 1),
                                          key_padding_mask=video_padding_mask)
        cross_att_out = cross_att_out.transpose(0, 1)
        # concat_out: (batch_size, seq_len, hidden_dim*2)
        concat_out = torch.cat([video_fc_out, cross_att_out], dim=-1)
        # pred: (batch_size, seq_len, 1)
        pred = self.fc(concat_out)
        return pred

class SpanExtraction(nn.Module):
    def __init__(self, dim=768):
        super(SpanExtraction, self).__init__()
        self.cross_att = nn.MultiheadAttention(dim, num_heads=8)
        self.q_v_att = nn.MultiheadAttention(dim, num_heads=8)
        self.conv1 = nn.Conv1d(in_channels=1536, out_channels=768, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=1536, out_channels=768, kernel_size=3, padding=1)
        self.Dropout = nn.Dropout(0.1)
        self.fc_start = nn.Linear(52, dim)
        self.fc_end = nn.Linear(52, dim)
        self.softmax = nn.Softmax(dim=-1)
        self.entropys = nn.CrossEntropyLoss()
        self.relu = nn.ReLU()


    def forward(self,q_feat, q_mask,sub_feat,sub_msk,ious=None,vfeats=None,vfeats_mask=None,start_tgt=None,end_tgt=None):
        # txt_feat: (batch_size, txt_seq_len, txt_dim)
        # attention_mask: (batch_size, txt_seq_len)

        sub_feat = self.Dropout(sub_feat) # [1,512,768]
        vfeats = self.Dropout(vfeats) # [1,768,768]

        # cross_att_out: (batch_size, seq_len, hidden_dim) [512, 1, 768]

        sf = sub_feat.transpose(0, 1)
        cross_att_out, _ = self.cross_att(sub_feat.transpose(0, 1),
                                            vfeats.transpose(0, 1),
                                            vfeats.transpose(0, 1),
                                            key_padding_mask=vfeats_mask)
        
        
        cross_att_out = cross_att_out.transpose(0, 1)


        # concat_out: (batch_size, seq_len, hidden_dim*2)
        concat_out = torch.cat([sub_feat, cross_att_out], dim=-1) # [1,512,1536]


        # 将[512,1536] 降维到[512,768]
        concat_out = self.conv1(concat_out.transpose(1,2)).transpose(1,2) # [1,512,768]
        q_v_att_out, _ = self.q_v_att(q_feat.transpose(0, 1),
                                            concat_out.transpose(0, 1),
                                            concat_out.transpose(0, 1)) # [52, 1, 768 ]
        
        q_v_att_out = q_v_att_out.transpose(0, 1) # [1, 52, 768]
        q_v_concat_out = torch.cat([q_feat, q_v_att_out], dim=-1) # [1, 52, 1536]
        q_v_concat_out = self.conv2(q_v_concat_out.transpose(1,2)).transpose(1,2)
        
        x = q_v_concat_out.transpose(1, 2)  # 将序列长度维度移动到第3维
        start = self.fc_start(x).squeeze(-1)  # [1, 768]
        end = self.fc_end(x).squeeze(-1)  # [1, 768]
        start_logits = self.softmax(start)  # [1, 768]
        end_logits = self.softmax(end)  # [1, 768]
        if start is not None and end is not None:
            start_loss = self.entropys(start_logits, start_tgt)
            end_loss = self.entropys(end_logits, end_tgt)
            return start_loss, end_loss
        return start_logits, end_logits


