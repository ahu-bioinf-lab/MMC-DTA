import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gep, global_max_pool as gmp, global_add_pool as gap
from torch_geometric.nn import GCNConv, SAGEConv

class Highway(nn.Module):
    def __init__(self, num_highway_layers, input_size):
        super(Highway, self).__init__()
        self.num_highway_layers = num_highway_layers
        self.non_linear = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(self.num_highway_layers)])
        self.linear = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(self.num_highway_layers)])
        self.gate = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(self.num_highway_layers)])
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        for layer in range(self.num_highway_layers):
            gate = torch.sigmoid(self.gate[layer](x))
            non_linear = F.relu(self.non_linear[layer](x))
            linear = self.linear[layer](x)
            x = gate * non_linear + (1 - gate) * linear
            x = self.dropout(x)
        return x

class DualInteract(nn.Module):

    def __init__(self, field_dim, embed_size, head_num, dropout=0.2):
        super(DualInteract, self).__init__()
        self.bit_wise_net = Highway(input_size=field_dim * embed_size,
                                    num_highway_layers=2)  # 原來是2，改成3了

    def forward(self, x):
        """
            x : batch, field_dim, embed_dim
        """
        b, f, e = x.shape
        bit_wise_x = self.bit_wise_net(x.reshape(b, f * e))
        m_x = bit_wise_x
        return m_x



# GINConv model
class MMCNet(torch.nn.Module):
    def __init__(self, n_output=1, num_features_mol=78, num_features_pro=33,
                 protein_MAX_LENGH=1200, protein_kernel=[3,5,7], drug_MAX_LENGH=100, drug_kernel=[3,5,7], conv=32,
                 char_dim=128, head_num=8, output_dim=128, dropout=0.2):

        super(MMCNet, self).__init__()

        # 1）以下是药的GATGCN
        self.output_dim = output_dim
        self.n_output = n_output
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.cconv1 = GCNConv(num_features_mol, num_features_mol)
        self.cconv2 = GCNConv(num_features_mol, num_features_mol * 2)
        self.cconv3 = GCNConv(num_features_mol * 2, 128)


        self.conv = conv
        self.head_num = head_num
        self.drug_kernel = drug_kernel
        self.protein_kernel = protein_kernel
        #####1) target---GCN
        self.pro_conv1 = SAGEConv(num_features_pro, num_features_pro, aggr='sum')  # 刚刚改成mean了
        self.pro_conv2 = SAGEConv(num_features_pro, num_features_pro * 2, aggr='sum')
        self.pro_conv3 = SAGEConv(num_features_pro * 2, 128, aggr='sum')

        # ### 3） CNN+批量归一化
        # 定义用于药物特征提取的CNN模块
        self.Drug_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=self.conv, kernel_size= self.drug_kernel[0]),
            # nn.ReLU(),
            nn.BatchNorm1d(self.conv),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2, kernel_size= self.drug_kernel[1]),
            # nn.ReLU(),
            nn.BatchNorm1d(self.conv*2),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=self.conv * 2, out_channels=self.conv * 4, kernel_size= self.drug_kernel[2]),
            nn.AdaptiveMaxPool1d(1)
        )
         # # 定义用于蛋白质特征提取的CNN模块
        self.Protein_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=self.conv, kernel_size = self.protein_kernel[0]),
            # nn.ReLU(),
            nn.BatchNorm1d(self.conv),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2, kernel_size = self.protein_kernel[1]),
            # nn.ReLU(),
            nn.BatchNorm1d(self.conv*2),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=self.conv * 2, out_channels=self.conv * 4, kernel_size = self.protein_kernel[2]),
            nn.AdaptiveMaxPool1d(1)
        )
        self.sofmax = nn.Softmax(dim=1)
        # 全局平均池化层将特征长度缩减到1
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        # 全局最大池化层将特征长度缩减到1
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        # self.attention = Attention(96)  # 初始化注意力机制

        proj_dim = 128  # 128
        self.feature_interact = DualInteract(field_dim=4, embed_size=proj_dim, head_num=8)  # 修改处 dim=3改成5

        #对比
        self.constrast = SupConLoss()
        # # FCN层
        self.dropout_rate = nn.Dropout(0.1)
        self.leaky_relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(128*4, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 1)



    def forward(self, data_mol,data_pro):
        mol_x, mol_edge_index, mol_batch = data_mol.x, data_mol.edge_index, data_mol.batch
        target_x, target_edge_index, target_weight, target_batch = data_pro.x, data_pro.edge_index, data_pro.edge_weight, data_pro.batch

        drug_input1 = data_mol.smiles_emb # (64,384)
        target_input1 = data_pro.fasta_emb # (64,1280)
        drug_input = drug_input1.unsqueeze(1) # 0:(1,64,384) 1:(64,1,384)
        target_input = target_input1.unsqueeze(1)  # (64,1,1280)


        # 1)以下是药物的GCN
        x = self.cconv1(mol_x, mol_edge_index)  # (439,32)
        x = self.relu(x)
        x = self.cconv2(x, mol_edge_index)
        x = self.relu(x)
        x = self.cconv3(x, mol_edge_index)  # (439, 128)
        # 对卷积后的小分子进行图的池化操作
        x = gmp(x, mol_batch)  # (16, 128)  # 明天将这里改成gap试试
        # # x = self.flat(x)  # (16, 256)


        # 2)以下是蛋白质的GCN
        xt = self.pro_conv1(target_x, target_edge_index)
        xt = self.relu(xt)
        xt= self.pro_conv2(xt, target_edge_index)
        xt = self.relu(xt)
        xt = self.pro_conv3(xt, target_edge_index)
        # xt = self.relu(xt)
        xt = gep(xt, target_batch)  # global pooling


        # 3）使用 CNN 提取药物和蛋白质的特征 (128,96,378)
        drugConv = self.Drug_CNNs(drug_input)
        drugConv=drugConv.view(-1, 128)

        proteinConv = self.Protein_CNNs(target_input)
        proteinConv = proteinConv.view(-1, 128)

        # 4) 对比学习
        drug_con = self.constrast(x, drugConv, labels=None, mask=None)
        pro_con = self.constrast(xt, proteinConv, labels=None, mask=None)


        con = drug_con + pro_con
        all_features = torch.stack([x, drugConv, xt, proteinConv],dim=1)  # 堆叠多个模态(128,4,96)
        all_features = self.feature_interact(all_features)

        # add some dense layers
        xc = self.fc1(all_features)
        xc = self.leaky_relu(xc)
        xc = self.dropout_rate(xc)
        xc = self.fc2(xc)
        xc = self.leaky_relu(xc)
        xc = self.dropout_rate(xc)
        out = self.out(xc)

        return out ,con

class SupConLoss(nn.Module):

    def __init__(self, temperature=0.5, scale_by_temperature=True):
        super(SupConLoss, self).__init__()
        self.temperature = temperature  # 参数，用于缩放相似度
        self.scale_by_temperature = scale_by_temperature  #是否将最终损失乘以温度系数

    def forward(self, features1, features2, labels=None, mask=None):
        features = torch.cat((features1,features2),0)  # 将两个特征张量在第一维度上拼接，形成新的特征张量 （256，128）
        features = F.normalize(features, p=2, dim=1)  # 对特征进行L2归一化
        batch_size = features.shape[0]  # 样本的总数，128*2
        batch = features1.shape[0]  # 单个批次大小 128

        # 关于labels参数
        if labels is not None and mask is not None:  # labels和mask不能同时定义值，因为如果有label，那么mask是需要根据Label得到的
            raise ValueError('Cannot define both `labels` and `mask`')
        if labels is None and mask is None:  # 如果没有labels，也没有mask，就是无监督学习，mask是对角线为1的矩阵，表示(i,i)属于同一类
            mask = torch.eye(batch, dtype=torch.float32).to(features.device)
        elif labels is not None:  # 如果给出了labels, mask根据label得到，两个样本i,j的label相等时，mask_{i,j}=1
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(features.device)
        else:
            mask = mask.float().to(features.device)

        # mask = 2 * mask - torch.ones_like(mask) 扩展mask的维度，使其匹配拼接后的特征
        mask = torch.cat((mask,mask), 1)
        mask = torch.cat((mask, mask), 0)  #(256,256) 对角线为1，其余为0

        # compute logits  # （256，256）计算两个样本之间的相似度（点乘），并除以温度参数来控制相似度的尺度，
        # 其中每个元素 (i, j) 表示第 i 个样本和第 j 个样本之间的相似度。
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)      # 计算相似度并缩放
        # for numerical stability （256，1）
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True) # （256，1）找到每个样本最大的相似度
        logits = anchor_dot_contrast - logits_max.detach()  # （256，256）
        # 从每个相似度值中减去对应行的最大值。这是为了避免计算 exp 时数值过大导致的溢出问题。
        # 这一步保证了 logits 中的最大值为 0，因此在后续计算中不会导致非常大的 exp(logits)
        # logits = anchor_dot_contrast
        exp_logits = torch.exp(logits) # （256，256）进行指数运算，值在0-1之间

        # 构建mask （256，256）只有0和1
        logits_mask = torch.ones_like(mask).to(exp_logits.device) - torch.eye(batch_size).to(exp_logits.device)
        positives_mask = (mask * logits_mask).to(logits_mask.device)  # 正样本对的位置 (256,256)一个全0矩阵+单位矩阵+单位矩阵+全0矩阵
        negatives_mask = (1. - mask).to(logits_mask.device)  # 负样本对的位置
            # （256，1）值基本上都是1  计算每行的正样本对数量（除去自身对），这里正样本只有另一个模态的样本
        num_positives_per_row = torch.sum(positives_mask, axis=1).to(logits_mask.device)  # 除了自己之外，正样本的个数
         # 计算分母：计算每行中所有负样本的指数相似度和正样本的指数相似度之和
        denominator = torch.sum(exp_logits * negatives_mask, axis=1, keepdims=True) + torch.sum(exp_logits * positives_mask, axis=1, keepdims=True)
            # （256，256）
        log_probs = logits - torch.log(denominator)  # 计算log概率
        if torch.any(torch.isnan(log_probs)): # 检查是否有Nan值
            raise ValueError("Log_prob has nan!")
        log_probs = torch.sum(       # 只选择正样本对的log概率，并按照正样本数量平均。
            log_probs * positives_mask, axis=1)[num_positives_per_row > 0] / num_positives_per_row[
                        num_positives_per_row > 0]

        # log_probs = torch.sum(
        #     (logits - torch.log(denominator)) * positives_mask, axis=1) / 1

        # loss
        loss = -log_probs  # 计算最终损失：取负
        if self.scale_by_temperature:  # 是否需要按照温度参数缩放损失
            loss *= self.temperature
        loss = loss.mean()  # 对所有样本求平均损失
        return loss

