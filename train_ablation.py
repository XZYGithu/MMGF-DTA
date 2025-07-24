from models.ablation import *
from utils import *
import pandas as pd
import torch.nn as nn
import numpy as np
from torch_geometric.loader import DataLoader
from lifelines.utils import concordance_index
import argparse
import os

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
    # 创建保存目录
    save_dir = f'results/{args.dataset}/ablation'
    os.makedirs(save_dir, exist_ok=True)
    
    # 数据加载部分
    dataset = args.dataset
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else "cpu")
    
    # 读取数据
    df_train = pd.read_csv(f'data/{dataset}/train.csv')
    df_test = pd.read_csv(f'data/{dataset}/test.csv')
    train_smile, train_seq, train_label = list(df_train['compound_iso_smiles']), list(df_train['target_sequence']), list(df_train['affinity'])
    test_smile, test_seq, test_label = list(df_test['compound_iso_smiles']), list(df_test['target_sequence']), list(df_test['affinity'])

    # 加载图数据
    with open(f"data/{dataset}/mol_data.pkl", 'rb') as file:
        mol_data = pickle.load(file)
    with open(f'data/{dataset}/pro_data.pkl', 'rb') as file2:
        pro_data = pickle.load(file2)
    with open(f'data/{dataset}/PPI/ppi_data.pkl', 'rb') as file3:
        ppi_adj, ppi_features, ppi_index = pickle.load(file3)

    # 数据转换
    ppi_adj = torch.LongTensor(np.argwhere(ppi_adj == 1).transpose(1, 0)).to(device)
    ppi_features = torch.Tensor(ppi_features).to(device)
    pro_graph = proGraph(pro_data, ppi_index, device)

    # 数据集与数据加载器
    fingerprint_path = "ligand_to_ecfp.pkl"  
    train_dataset = DTADataset(train_smile, train_seq, train_label, mol_data=mol_data, ppi_index=ppi_index, smiles=train_smile)
    test_dataset = DTADataset(test_smile, test_seq, test_label, mol_data=mol_data, ppi_index=ppi_index, smiles=test_smile)
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, collate_fn=collate, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, collate_fn=collate, num_workers=args.num_workers)

    # 模型初始化
    model = AblationFactory.create_model(
        model_type=args.model_type,
        ablation_type=args.ablation,
        n_output=1,
        output_dim=128,
        num_features_xd=78,
        num_features_pro=33,
        num_features_ppi=1442,
        fingerprint_path=fingerprint_path
    ).to(device)

    # 训练配置
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.LR)
    best_mse = 1000
    best_ci = 0  
    best_rm2 = -1  
    best_epoch = -1  
    model_file = f'{save_dir}/model_{args.ablation}.pt'
    result_file = f'{save_dir}/results_{args.ablation}.csv'

    # 开始训练
    for epoch in range(args.epochs):
        train(model, device, train_loader, optimizer, ppi_adj, ppi_features, pro_graph, loss_fn, args, epoch + 1)  # 训练
        G, P = test(model, device, test_loader, ppi_adj, ppi_features, pro_graph)  # 测试
        # 计算MSE CI RM2
        rm2 = calculate_rm2(G, P)
        ret = [mse(G, P), concordance_index(G, P), rm2]
        
        if ret[0] < best_mse:  
            torch.save(model.state_dict(), model_file)
            with open(result_file, 'w') as f:
                f.write(','.join(map(str, ret)))
            best_epoch = epoch + 1
            best_mse = ret[0]
            best_ci = ret[1]
            best_rm2 = ret[2]
            print(f'{dataset}: mse improved at epoch :{best_epoch} ; best_mse,best_ci,best_rm²: {best_mse:.4f}, {best_ci:.4f}, {best_rm2:.4f}')
        else:
            print(f'{dataset}: mse No improvement since epoch :{best_epoch} ; best_mse,best_ci,best_rm²:  {best_mse:.4f}, {best_ci:.4f}, {best_rm2:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='default', 
                        choices=['default', 'gcn', 'graphsage'],
                        help='模型架构类型: default(混合架构), gcn(纯GCN), graphsage(纯GraphSAGE)')
    parser.add_argument('--ablation', type=str, default='no_ppi', choices=['no_coatt', 'no_ppi', 'no_fp'])
    parser.add_argument('--dataset', type=str, default='davis', choices=['davis','kiba'])
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--LR', type=float, default=0.0005)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--log_interval', type = int, default = 20)  # 打印日志的间隔
    parser.add_argument('--k_dim', type=int, default=64)  # MutualAttention的k维度
    parser.add_argument('--beta_x', type=float, default=0.5)  # 残差连接权重
    parser.add_argument('--beta_p', type=float, default=0.5)
    args = parser.parse_args()
    main(args)