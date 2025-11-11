import os
from torch_geometric.data import InMemoryDataset, DataLoader, Batch
from torch_geometric import data as DATA
import torch
from tqdm import tqdm
import numpy as np
from math import sqrt
from scipy import stats
from sklearn.metrics import r2_score

# initialize the dataset
class DTADataset(InMemoryDataset):
    def __init__(self, root='data', dataset='Parasite',
                 xd=None,smi_char=None, xt_char=None, smi_emb=None, fas_emb=None, y=None, transform=None,
                 pre_transform=None, target_key=None, smile_graph=None,  target_graph=None): #, clique_graph=None,):

        super(DTADataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.xd = xd
        self.smi_char = smi_char
        self.xt_char = xt_char
        self.smi_emb = smi_emb
        self.fas_emb = fas_emb
        self.y = y
        self.target_key = target_key
        self.smile_graph = smile_graph
        self.target_graph = target_graph

        self.process(xd, smi_char, xt_char, smi_emb, fas_emb, y, target_key, smile_graph, target_graph)

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + '_data_mol.pt', self.dataset + '_data_pro.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self): # 创建处理后文件目录
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

     # 处理输入的数据集，将其转为图数据
    def process(self, xd, smi_char, xt_char, smi_emb, fas_emb, y, target_key, smile_graph, target_graph):
        # 这里是训练集、测试集中的药物smiles序列和氨基酸序列和标签
        assert (len(xd) == len(xt_char) and len(xt_char) == len(y)), 'The three lists must have the same length!'

        data_list_mol = []
        data_list_pro = []
        # data_list_clique = []
        data_len = len(xd) # 数据集样本数量
        print('loading tensors ...')
        for i in tqdm(range(data_len)):
            print('Converting to graph: {}/{}'.format(i + 1, data_len))
            smiles = xd[i] # 获取数据集中的药物和蛋白质
            d_char = smi_char[i]
            t_char = xt_char[i]
            drug_emb = smi_emb[smiles]
            t_key = target_key[i]
            target_emb = fas_emb[t_key]
            labels = y[i]

             # 从图数据字典中获取药物分子图和蛋白质图
            mol_size, mol_features, mol_edge_index = smile_graph[smiles]
            # 创建药物图数据
            GCNData_mol = DATA.Data(x=torch.Tensor(mol_features),
                                    edge_index=torch.LongTensor(mol_edge_index).transpose(1, 0),
                                    y=torch.FloatTensor([labels]))
            GCNData_mol.smiles_char = torch.LongTensor([d_char])
            GCNData_mol.smiles_emb = torch.Tensor([drug_emb])
            GCNData_mol.__setitem__('c_size', torch.LongTensor([mol_size]))

            target_size, target_features, target_edge_index, target_edge_weight = target_graph[t_key]
            # 创建蛋白质图数据
            GCNData_pro = DATA.Data(x=torch.Tensor(target_features),
                                    edge_index=torch.LongTensor(target_edge_index).transpose(1, 0),
                                    edge_weight=torch.FloatTensor(target_edge_weight),
                                    y=torch.FloatTensor([labels]))
            GCNData_pro.target_char = torch.LongTensor([t_char])
            GCNData_pro.fasta_emb = torch.Tensor([target_emb])
            GCNData_pro.__setitem__('target_size', torch.LongTensor([target_size]))

             # 将图数据添加到对应列表中
            data_list_mol.append(GCNData_mol)
            data_list_pro.append(GCNData_pro)
            # data_list_clique.append(GCNData_clique)

         # 过滤图数据
        if self.pre_filter is not None:
            data_list_mol = [data for data in data_list_mol if self.pre_filter(data)]
            data_list_pro = [data for data in data_list_pro if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list_mol = [self.pre_transform(data) for data in data_list_mol]
            data_list_pro = [self.pre_transform(data) for data in data_list_pro]
        print('Graph construction done. Saving to file.')

        # 将处理后的图数据对象列表赋值给类属性
        self.data_mol = data_list_mol
        self.data_pro = data_list_pro


    def __len__(self):
        return len(self.data_mol)

    def __getitem__(self, idx):

        return self.data_mol[idx], self.data_pro[idx]



def rmse(y, f):
    rmse = sqrt(((y - f) ** 2).mean(axis=0))
    return rmse


def mse(y, f):
    mse = ((y - f) ** 2).mean(axis=0)
    return mse


def pearson(y, f):
    rp = np.corrcoef(y, f)[0, 1]
    return rp


def spearman(y, f):
    rs = stats.spearmanr(y, f)[0]
    return rs

def CI2(y,f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci

def ci(gt, pred):
    gt_mask = gt.reshape((1, -1)) > gt.reshape((-1, 1))
    diff = pred.reshape((1, -1)) - pred.reshape((-1, 1))
    h_one = (diff > 0)
    h_half = (diff == 0)
    CI = np.sum(gt_mask * h_one * 1.0 + gt_mask * h_half * 0.5) / np.sum(gt_mask)

    return CI

def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))

    return mult / float(y_obs_sq * y_pred_sq)


def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs * y_pred) / float(sum(y_pred * y_pred))


def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k * y_pred)) * (y_obs - (k * y_pred)))
    down = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))

    return 1 - (upp / float(down))


def get_rm2(ys_orig, ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)

    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))

class BestMeter(object):
    """Computes and stores the best value"""
    def __init__(self, best_type):
        self.best_type = best_type
        self.count = 0
        self.reset()

    def reset(self):
        if self.best_type == 'min':
            self.best = float('inf')
        else:
            self.best = -float('inf')

    def update(self, best):
        self.best = best
        self.count = 0

    def get_best(self):
        return self.best

    def counter(self):
        self.count += 1
        return self.count

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def get_average(self):
        self.avg = self.sum / (self.count + 1e-12)

        return self.avg

def save_model_dict(model, model_dir, msg):
    model_path = os.path.join(model_dir, msg + '.pt')
    torch.save(model.state_dict(), model_path)
    print("model has been saved to %s." % (model_path))

def load_model_dict(model, ckpt):
    model.load_state_dict(torch.load(ckpt))

def collate(data_list):
    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])
    batchC = Batch.from_data_list([data[2] for data in data_list])
    return batchA, batchB, batchC
