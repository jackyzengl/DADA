import torch
import torch.nn as nn
import torch.nn.functional as F


class GAT_layer(nn.Module):
    def __init__(self, 
                 head=1, 
                 dim_in=64, headdim_out=64, 
                 bias=True,):
        super(GAT_layer, self).__init__()
        self.head = head
        self.dim_in, self.headdim_out = dim_in, headdim_out
        self.bias = bias
        
        self.w = nn.Parameter(torch.Tensor(head, dim_in, headdim_out))  # [head, dim_in, headdim_out]
        self.a_src = nn.Parameter(torch.Tensor(head, headdim_out, 1))  # [head, headdim_out, 1]
        self.a_dst = nn.Parameter(torch.Tensor(head, headdim_out, 1))  # [head, headdim_out, 1]

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        
        if bias:
            self.b = nn.Parameter(torch.Tensor(headdim_out))  # [headdim_out]
            nn.init.constant_(self.b, 0)

        nn.init.xavier_uniform_(self.w, gain=1.414)
        nn.init.xavier_uniform_(self.a_src, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst, gain=1.414)

    def forward(self, 
                h):  # [len, nums_i, dim_in=64|32]
        n = h.shape[1]
        h_ = torch.matmul(h.unsqueeze(dim=1), self.w)  # [len, head, nums_i, headdim_out=32|64]
        attn_src = torch.matmul(h_, self.a_src)  # [len, head, nums_i, 1]
        attn_dst = torch.matmul(h_, self.a_dst)  # [len, head, nums_i, 1]
        attn = attn_src.expand(-1, -1, -1, n) +\
               attn_dst.expand(-1, -1, -1, n).permute(0, 1, 3, 2)  # [len, head, nums_i, nums_i]

        attn = self.leaky_relu(attn)
        attn = self.softmax(attn)  # [len, head, nums_i, nums_i]
        h_out = torch.matmul(attn, h_)  # [len, head, nums_i, headdim_out]
        if self.bias:
            return h_out + self.b  # [len, head, nums_i, headdim_out]
        else:
            return h_out  # [len, head, nums_i, headdim_out]


class GAT_layers(nn.Module):
    def __init__(self, 
                 headdim_list=[64, 32, 64], 
                 head_list=[4, 1],):
        super(GAT_layers, self).__init__()
        self.n_layer = len(headdim_list) - 1  # 2
        self.gat_layers = nn.ModuleList()
        for i in range(self.n_layer):
            if i == 0:
                dim_in = headdim_list[i]
            else:
                dim_in = headdim_list[i] * head_list[i-1]
            self.gat_layers.append(GAT_layer(head=head_list[i],
                                             dim_in=dim_in, headdim_out=headdim_list[i+1]))
        self.norm_list = [torch.nn.InstanceNorm1d(64),
                          torch.nn.InstanceNorm1d(128),]

    def forward(self, 
                x):  # [len, nums_i, 64]
        length, n = x.shape[:2]
        for i, gat_layer in enumerate(self.gat_layers):
            x = self.norm_list[i](x.permute(0, 2, 1)).permute(0, 2, 1)  # [len, nums_i, dim_in=64]
            x = gat_layer(x)  # [len, head=4, nums_i, headdim_out=32]
            
            if i+1 == self.n_layer:  # the last GAT_layer
                x = x.squeeze(dim=1)  # [len, nums_i, 64] 
            else:  # the mid GAT_layers
                #x = F.elu(x.transpose(1, 2).contiguous().view(length, n, -1))  # [len, nums_i, 32*4]
                x = F.elu(x.permute(0, 2, 1, 3).contiguous().view(length, n, -1))  # [len, nums_i, 32*4]
        else:
            return x  # [len, nums_i, 64]


class GAT_module(nn.Module):
    def __init__(self, 
                 headdim_list=[64, 32, 64], 
                 head_list=[4, 1],):
        super(GAT_module, self).__init__()
        self.gat_net = GAT_layers(headdim_list, head_list)

    def forward(self, 
                in_features,  # [nums, 64, len]
                seq_start_end,):  # [(start0, end0), (start1, end1), ...])
        gat_features = []
        for start_i, end_i in seq_start_end.data:
            in_feature_i = in_features[start_i:end_i, :, :]  # [nums_i, 64, len]
            in_feature_i_ = in_feature_i.permute(2, 0, 1)  # [len, nums_i, 64]
            gat_feature_i = self.gat_net(in_feature_i_)  # [len, nums_i, 64]
            gat_features.append(gat_feature_i)
            
        gat_features = torch.cat(gat_features, dim=1)  # [len, nums, 64]
        gat_features_ = gat_features.permute(1, 2, 0)  # [nums, 64, len]
        return gat_features_  # [nums, 64, len]


class Mean_module(nn.Module):
    """对每帧的每个agent，其64维特征向量求平均，在repeat 64次，作为新的特征向量"""
    def __init__(self, 
                 input_dim=64):
        super(Mean_module, self).__init__()
        self.input_dim = input_dim

    def forward(self, 
                in_features,  # [nums, 64, len]
                seq_start_end,):  # [(start0, end0), (start1, end1), ...])
        out_features = []
        for start_i, end_i in seq_start_end.data:
            in_feature_i = in_features[start_i:end_i, :, :]  # [nums_i, 64, len]
            in_feature_i_ = in_feature_i.permute(2, 0, 1)  # [len, nums_i, 64]
            mean_in_feature = torch.mean(in_feature_i_, dim=2).unsqueeze(dim=-1)  # [len, nums_i, 1]
            out_feature_i = mean_in_feature.repeat(1, 1, self.input_dim)  # [len, nums_i, 64]
            out_features.append(out_feature_i)
            
        out_features = torch.cat(out_features, dim=1)  # [len, nums, 64]
        out_features_ = out_features.permute(1, 2, 0)  # [nums, 64, len]
        return out_features_  # [nums, 64, len]
