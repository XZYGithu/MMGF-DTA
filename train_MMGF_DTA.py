from models.MMGF import *
from utils import *
import pandas as pd
import numpy as np
from torch_geometric.loader import DataLoader
from lifelines.utils import concordance_index
import argparse


def train(model, device, train_loader, optimizer, ppi_adj, ppi_features, pro_graph, loss_fn, args, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        mol_data = data[0].to(device)
        pro_data = data[1].to(device)
        mol_smiles = data[2]  # SMILES in the data
        optimizer.zero_grad()
            # 前向传播包含分子指纹处理
        output = model(mol_data, pro_data, ppi_adj, ppi_features, pro_graph, mol_smiles)
        loss = loss_fn(output, mol_data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * args.batch,
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))

def test(model, device, loader, ppi_adj, ppi_features, pro_graph):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            mol_data = data[0].to(device)
            pro_data = data[1].to(device)
            mol_smiles = data[2]  # SMILES in the data
            output = model(mol_data, pro_data, ppi_adj, ppi_features, pro_graph, mol_smiles)
            total_preds = torch.cat((total_preds, output.cpu()), 0) #predicted values
            total_labels = torch.cat((total_labels, mol_data.y.view(-1, 1).cpu()), 0) #ground truth
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()

def main(args):
    dataset = args.dataset
    model_dict_ = {'MMGF': MMGF}  
    modeling = model_dict_[args.model]  
    model_st = modeling.__name__
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else "cpu")  

    # 读取训练和测试数据
    df_train = pd.read_csv(f'data/{dataset}/train.csv')  # 读取训练数据
    df_test = pd.read_csv(f'data/{dataset}/test.csv')  # 读取测试数据
    train_smile, train_seq, train_label = list(df_train['compound_iso_smiles']), list(df_train['target_sequence']), list(df_train['affinity'])
    test_smile, test_seq, test_label = list(df_test['compound_iso_smiles']), list(df_test['target_sequence']), list(df_test['affinity'])

    # 读取其他图数据（药物图、蛋白质图、PPI图）
    with open(f"data/{dataset}/mol_data.pkl", 'rb') as file:
        mol_data = pickle.load(file)  # 读取药物图数据
    with open(f'data/{dataset}/pro_data.pkl', 'rb') as file2:
        pro_data = pickle.load(file2)  # 读取蛋白质图数据
    with open(f'data/{dataset}/PPI/ppi_data.pkl', 'rb') as file3:
        ppi_adj, ppi_features, ppi_index = pickle.load(file3)  # 读取PPI图数据

    # 将PPI图和特征转换为PyTorch张量
    ppi_adj = torch.LongTensor(np.argwhere(ppi_adj == 1).transpose(1, 0)).to(device)  # PPI邻接矩阵转为张量
    ppi_features = torch.Tensor(ppi_features).to(device)  # PPI特征矩阵转为张量
    pro_graph = proGraph(pro_data, ppi_index, device)  # 创建蛋白质图

    # 设置分子指纹文件路径并传递给模型
    fingerprint_path = f"ligand_to_ecfp.pkl"  
    train_dataset = DTADataset(train_smile, train_seq, train_label, mol_data=mol_data, ppi_index=ppi_index, smiles=train_smile)
    test_dataset = DTADataset(test_smile, test_seq, test_label, mol_data=mol_data, ppi_index=ppi_index, smiles=test_smile)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, collate_fn=collate, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, collate_fn=collate, num_workers=args.num_workers)

    # 实例化模型，并将分子指纹路径传递给模型
    model = modeling(fingerprint_path=fingerprint_path).to(device)  # 将fingerprint_path传递给模型

    loss_fn = nn.MSELoss()  # 损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=args.LR)  # Adam优化器
    best_mse = 1000  
    best_ci = 0  
    best_rm2 = -1  
    best_epoch = -1  
    model_file_name = f'results/{dataset}/' + f'train_{model_st}aaaaa.model'  # 模型保存路径
    result_file_name = f'results/{dataset}/' + f'train_{model_st}aaaa.csv'  # 结果保存路径
    
    # 开始训练
    for epoch in range(args.epochs):
        train(model, device, train_loader, optimizer, ppi_adj, ppi_features, pro_graph, loss_fn, args, epoch + 1)  # 训练
        G, P = test(model, device, test_loader, ppi_adj, ppi_features, pro_graph)  # 测试
        # 计算MSE CI RM2
        rm2 = calculate_rm2(G, P)
        ret = [mse(G, P), concordance_index(G, P), rm2]
        
        if ret[0] < best_mse:  
            torch.save(model.state_dict(), model_file_name)
            with open(result_file_name, 'w') as f:
                f.write(','.join(map(str, ret)))
            best_epoch = epoch + 1
            best_mse = ret[0]
            best_ci = ret[1]
            best_rm2 = ret[2]
            print(f'{dataset}: mse improved at epoch :{best_epoch} ; best_mse,best_ci,best_rm²: {best_mse:.4f}, {best_ci:.4f}, {best_rm2:.4f}')
        else:
            print(f'{dataset}: mse No improvement since epoch :{best_epoch} ; best_mse,best_ci,best_rm²:  {best_mse:.4f}, {best_ci:.4f}, {best_rm2:.4f}')

# 主函数入口
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = str, default = 'MMGF')  # 选择模型类型
    parser.add_argument('--epochs', type = int, default = 1500)  # 设置训练轮次
    parser.add_argument('--batch', type = int, default = 512)  # 设置批次大小
    parser.add_argument('--LR', type = float, default = 0.0005)  # 学习率
    parser.add_argument('--log_interval', type = int, default = 20)  # 打印日志的间隔
    parser.add_argument('--device', type = int, default = 1)  # 使用的GPU设备编号
    parser.add_argument('--dataset', type = str, default = 'davis',choices = ['davis','kiba'])  # 数据集选择
    parser.add_argument('--num_workers', type= int, default = 10)  # 数据加载时的工作线程数
    args = parser.parse_args()  # 解析命令行参数
    main(args)  # 调用主函数




