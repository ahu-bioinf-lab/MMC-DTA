# -*-coding:utf-8-*-
# -*-coding:utf-8-*-
import time
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 调试用
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader
from dataloader import create_DTA_dataset
from model_contrastive import MMCNet
from utils import *
import logging


def train(model, device, train_loader, optimizer, loss_fn):
    train_losses_in_epoch = []  # 训练阶段
    model.train()
    for data in train_loader:
        '''data preparation '''  # 数据准备
        data_mol = data[0].to(device)
        data_pro = data[1].to(device)

        optimizer.zero_grad()  # 梯度清零

        output, con = model.forward(data_mol, data_pro)  # 前向传播
        train_loss = loss_fn(output, data_mol.y.view(-1, 1).float().to(device)) + con# 计算损失
        train_losses_in_epoch.append(train_loss.item())

        train_loss.backward()  # 反向传播
        optimizer.step()

    # 计算并记录一个epoch的平均训练损失
    train_loss_a_epoch = np.average(train_losses_in_epoch)

    return train_loss_a_epoch

def test(dataloader, model, loss_fn):
    valid_losses_in_epoch = []
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    with torch.no_grad():
        for data in dataloader:
            data_mol = data[0].to(device)
            data_pro = data[1].to(device)
            output,con = model.forward(data_mol, data_pro)  # 前向传播
            val_loss = loss_fn(output, data_mol.y.view(-1, 1).float().to(device))  # 计算损失
            valid_losses_in_epoch.append(val_loss.item())
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels,data_mol.y.view(-1, 1).cpu()), 0)
        valid_loss_a_epoch = np.average(valid_losses_in_epoch)

    return valid_loss_a_epoch, total_labels.numpy().flatten(), total_preds.numpy().flatten()

datasets = ['Parasite', 'Davis', 'Kiba']
Folds = ['fold_1','fold_2','fold_3','fold_4','fold_5']


model_st = 'MMC-DTA-Parasite'
logging.basicConfig(filename=f'{model_st}.log', level=logging.DEBUG)
BATCH_SIZE = 16
device = torch.device("cpu")
LR = 5e-5

all_mse, all_rmse, all_mae, all_rm2, all_ci, all_spearman, all_pearson= [], [], [], [], [], [],[]
# weight_decay = 1e-4
# Learning_rate = 5e-5
# Patience = 50

for n, fold in enumerate(Folds, start=1):
    print('\nrunning on ', model_st + '_' + fold)
    train_data, val_data, test_data = create_DTA_dataset(datasets[0], fold)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    save_path_i = "models_five/{}_Fold/".format(n)
    if not os.path.exists(save_path_i):
        os.makedirs(save_path_i)
    # 重新初始化模型、损失函数和优化器
    model = MMCNet().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    epochs = 1000
    best_mse = 100
    patience = 0  # 耐心计数器，用于早停
    Patience = 100
    print('第---%d---折训练开始'%(n))
    logging.info(f'第---{n}---折训练开始')
    for epoch in range(epochs):
        train_loss = train(model, device, train_loader, optimizer, loss_fn)
        val_loss, G, P = test(val_loader, model, loss_fn)
        val_mse = mse(G, P)
        val_mae = mean_absolute_error(G, P)
        val_person = pearson(G, P)

        print('epoch-%d, train_loss-%.3f, val_loss-%.3f, val_mse-%.3f, val_mae-%.3f, val_person-%.3f' % (
            epoch + 1, train_loss, val_loss, val_mse, val_mae, val_person))

        if val_mse < best_mse:
            best_mse = val_mse
            patience = 0
            torch.save(model.state_dict(), save_path_i + model_st + '-valid_best_model.pth')
        else:
            patience += 1

        if patience == Patience:  # 如果耐心计数器达到设定值，停止训练
            break

    # 保存最终模型
    torch.save(model.state_dict(), save_path_i + model_st + '-stable_model.pth')
    """load trained model"""  # 加载最佳模型
    model.load_state_dict(torch.load(save_path_i + model_st + "-valid_best_model.pth"))

    test_loss, G, P = test(test_loader, model, loss_fn)
    # 将预测结果和真实标签保存到CSV文件
    # df = pd.DataFrame({
    #     'True Labels': G,
    #     'Predicted Outputs': P
    # })
    # df.to_csv('Progesterone-test.csv', index=False)
    # print(f"预测结果已保存")
    test_mse, test_rmse, test_mae, test_person, test_sperman, test_ci, test_rm2 = mse(G, P), rmse(G,P), mean_absolute_error(G, P), pearson(G, P), spearman(G, P), CI2(G, P), get_rm2(G, P)
    print('test_mse-%.4f, test_rmse-%.4f,test_mae-%.4f, test_ci-%.4f, test_rm2-%.4f' % (test_mse, test_rmse, test_mae, test_ci, test_rm2))
    logging.info(f'Test mse {test_mse}, rmse {test_rmse}, mae {test_mae}, person {test_person}, sperman {test_sperman}, ci{test_ci}, rm2 {test_rm2}')
    logging.info(f'第---{n}---折训练完成')
    print('第---%d---折训练完成' % (n))
    all_mse.append(test_mse)
    all_rmse.append(test_rmse)
    all_mae.append(test_mae)
    all_pearson.append(test_person)
    all_spearman.append(test_sperman)
    all_ci.append(test_ci)
    all_rm2.append(test_rm2)


# 计算平均值
average_mse = np.mean(all_mse)
average_rmse = np.mean(all_rmse)
average_mae = np.mean(all_mae)
average_pearson = np.mean(all_pearson)
average_spearman = np.mean(all_spearman)
average_ci = np.mean(all_ci)
average_rm2 = np.mean(all_rm2)

print(f'五折平均 MSE: {average_mse:.4f}')
print(f'五折平均 RMSE: {average_rmse:.4f}')
print(f'五折平均 MAE: {average_mae:.4f}')
print(f'五折平均 Pearson: {average_pearson:.4f}')
print(f'五折平均 Spearson: {average_spearman:.4f}')
print(f'五折平均 CI: {average_ci:.4f}')
print(f'五折平均 RM2: {average_rm2:.4f}')

logging.info(f'The results on five_data:')
logging.info(f'average_mse :{average_mse}')
logging.info(f'average_rmse :{average_rmse}')
logging.info(f'average_mae :{average_mae}')
logging.info(f'average_pearson :{average_pearson}')
logging.info(f'average_spearman :{average_spearman}')
logging.info(f'average_ci :{average_ci}')
logging.info(f'average_rm2 :{average_rm2}')