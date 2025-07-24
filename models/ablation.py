import pickle
import torch
import torch.nn as nn
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GCNConv, GraphSAGE, global_mean_pool as gep
import numpy as np
import torch.nn.functional as F

class CoAttention(nn.Module):
    def __init__(self, drug_dim, protein_dim, k=64):
        super().__init__()
        self.k = k
        
        # 药物到蛋白质的注意力参数
        self.W_q_x = nn.Linear(drug_dim, k)  # 药物查询变换
        self.W_k_p = nn.Linear(protein_dim, k)  # 蛋白质键变换
        self.W_v_p = nn.Linear(protein_dim, k)  # 蛋白质值变换
        
        # 蛋白质到药物的注意力参数
        self.W_q_p = nn.Linear(protein_dim, k)  # 蛋白质查询变换
        self.W_k_x = nn.Linear(drug_dim, k)  # 药物键变换
        self.W_v_x = nn.Linear(drug_dim, k)  # 药物值变换
        
        # 输出变换
        self.fc_x = nn.Linear(k, drug_dim)
        self.fc_p = nn.Linear(k, protein_dim)
        
        # 初始化参数
        nn.init.xavier_normal_(self.W_q_x.weight)
        nn.init.xavier_normal_(self.W_k_p.weight)
        nn.init.xavier_normal_(self.W_v_p.weight)
        nn.init.xavier_normal_(self.W_q_p.weight)
        nn.init.xavier_normal_(self.W_k_x.weight)
        nn.init.xavier_normal_(self.W_v_x.weight)
        nn.init.xavier_normal_(self.fc_x.weight)
        nn.init.xavier_normal_(self.fc_p.weight)

    def forward(self, x, p):
        """
        x: 药物特征 [batch_size, drug_dim]
        p: 蛋白质特征 [batch_size, protein_dim]
        返回融合后的联合特征
        """
        # 药物到蛋白质的注意力
        Q_x = self.W_q_x(x)  # [B, k]
        K_p = self.W_k_p(p)  # [B, k]
        V_p = self.W_v_p(p)  # [B, k]
        
        # 计算注意力得分
        scores_x = torch.bmm(Q_x.unsqueeze(1), K_p.unsqueeze(2))  
        scores_x = scores_x / (self.k ** 0.5)
        alpha_x = F.softmax(scores_x, dim=-1)
        
        # 蛋白质特征加权
        attended_p = alpha_x * V_p.unsqueeze(1)  
        attended_p = attended_p.squeeze(1)       
        attended_p = self.fc_p(attended_p)       

        # 蛋白质到药物的注意力
        Q_p = self.W_q_p(p)  # [B, k]
        K_x = self.W_k_x(x)  # [B, k]
        V_x = self.W_v_x(x)  # [B, k]
        
        # 计算注意力得分
        scores_p = torch.bmm(Q_p.unsqueeze(1), K_x.unsqueeze(2))  # [B, 1, 1]
        scores_p = scores_p / (self.k ** 0.5)
        alpha_p = F.softmax(scores_p, dim=-1)
        
        # 药物特征加权
        attended_x = alpha_p * V_x.unsqueeze(1)  # [B, 1, k]
        attended_x = attended_x.squeeze(1)       # [B, k]
        attended_x = self.fc_x(attended_x)       # [B, drug_dim]

        # 拼接融合特征
        fused = torch.cat([attended_x, attended_p], dim=1)  # [B, drug_dim+protein_dim]
        return fused


class AblationMMGF(nn.Module):
    def __init__(self, ablation=None, n_output=1, output_dim=128, num_features_xd=78, 
                num_features_pro=33, num_features_ppi=1442, fingerprint_path=None, fp_dim=2048):
        super().__init__()
        self.ablation = ablation
        self.output_dim = output_dim
        self.n_output = n_output
        self.fp_dim = fp_dim

        # 加载分子指纹数据
        if fingerprint_path is not None:
            with open(fingerprint_path, 'rb') as f:
                self.fingerprints = pickle.load(f)
        else:
            self.fingerprints = {}

        # 药物图特征提取
        self.molGconv1 = GCNConv(num_features_xd, num_features_xd*2)
        self.molGconv2 = GraphSAGE(num_features_xd*2, num_features_xd*4, num_layers=1, aggr='mean')
        self.molGconv3 = GCNConv(num_features_xd*4, output_dim)
        self.molGconv4 = GraphSAGE(output_dim, output_dim, num_layers=1, aggr='mean')
        self.molFC1 = nn.Linear(output_dim, 1024)
        self.molFC2 = nn.Linear(1024, output_dim)

        # 蛋白质特征提取
        self.proGconv1 = GCNConv(num_features_pro, output_dim)
        self.proGconv2 = GraphSAGE(output_dim, output_dim, num_layers=1, aggr='mean')
        self.proGconv3 = GCNConv(output_dim, output_dim)
        self.proFC1 = nn.Linear(output_dim, 1024)
        self.proFC2 = nn.Linear(1024, output_dim)

        # PPI特征提取
        if ablation != "no_ppi":
            self.ppiGconv1 = GCNConv(num_features_ppi, 1024)
            self.ppiGconv2 = GraphSAGE(1024, 512, num_layers=1, aggr='mean')
            self.ppiGconv3 = GCNConv(512, output_dim)
            self.ppiGconv4 = GraphSAGE(output_dim, output_dim, num_layers=1, aggr='mean')
            self.ppiFC1 = nn.Linear(output_dim, 1024)
            self.ppiFC2 = nn.Linear(1024, output_dim)
            self.pro_combine_fc = nn.Linear(256, 128)

        # 分子指纹处理
        if ablation != "no_fp":
            self.fp_lstm = nn.LSTM(1, output_dim, batch_first=True)
            self.fp_fc = nn.Linear(output_dim, output_dim)

        # 特征融合模块
        if self.ablation != "no_coatt":
            # 动态调整 drug_dim
            drug_dim = output_dim if self.ablation == "no_fp" else output_dim * 2
            self.mutual_att = CoAttention(drug_dim, output_dim, 64)
            
            fc1_input_dim = drug_dim + output_dim
        else:
            self.fc_concat = nn.Linear(output_dim * 3, 384)
            fc1_input_dim = 384  # 直接拼接后的维度

        # 公共分类器
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(fc1_input_dim, 1024)  # 动态设置输入维度
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, n_output)

    def forward(self, mol_data, pro_data, ppi_edge, ppi_features, pro_graph, mol_smiles=None):
        x, edge_index, batch = mol_data.x, mol_data.edge_index, mol_data.batch
        seq_num = pro_data.seq_num
        p_x, p_edge_index, p_edge_len, p_batch = pro_graph

        # 药物特征提取
        x = self.relu(self.molGconv1(x, edge_index))
        x = self.relu(self.molGconv2(x, edge_index))
        x = self.relu(self.molGconv3(x, edge_index))
        x = self.relu(self.molGconv4(x, edge_index))
        x = gep(x, batch)  
        x = self.dropout2(self.relu(self.molFC1(x)))
        mol_feature = self.dropout2(self.molFC2(x))

        # 分子指纹处理
        if self.ablation != "no_fp" and mol_smiles is not None:
            fp_list = [self.fingerprints.get(smile, np.zeros(self.fp_dim)) for smile in mol_smiles]
            fp_tensor = torch.tensor(np.array(fp_list), dtype=torch.float32).to(x.device)
            fp_input = fp_tensor.unsqueeze(-1)
            lstm_out, (hidden, cell) = self.fp_lstm(fp_input)
            fp_feature = self.fp_fc(hidden.squeeze(0))
            drug_feature = torch.cat([mol_feature, fp_feature], dim=1)
        else:
            drug_feature = mol_feature

        # 蛋白质特征提取
        p_x = self.relu(self.proGconv1(p_x, p_edge_index))
        p_x = self.relu(self.proGconv2(p_x, p_edge_index))
        p_x = self.relu(self.proGconv3(p_x, p_edge_index))
        p_x = gep(p_x, p_batch)
        p_x = self.dropout2(self.relu(self.proFC1(p_x)))
        pro_onefeature = self.dropout2(self.proFC2(p_x))

        # PPI处理
        if self.ablation != "no_ppi":
            ppi_edge, _ = dropout_adj(ppi_edge, p=0.6, training=self.training)
            ppi_x = self.relu(self.ppiGconv1(ppi_features, ppi_edge))
            ppi_x = self.relu(self.ppiGconv2(ppi_x, ppi_edge))
            ppi_x = self.relu(self.ppiGconv3(ppi_x, ppi_edge))
            ppi_x = self.relu(self.ppiGconv4(ppi_x, ppi_edge))
            ppi_x = self.dropout1(self.relu(self.ppiFC1(ppi_x)))
            ppi_dualfeature = self.dropout1(self.ppiFC2(ppi_x))[seq_num]
            pro_selected = pro_onefeature[seq_num]
            pro_feature = torch.cat([pro_selected, ppi_dualfeature], dim=1)
            pro_feature = self.pro_combine_fc(pro_feature)
        else:
            pro_feature = pro_onefeature[seq_num]

        # 特征融合
        if self.ablation != "no_coatt":
            combined = self.mutual_att(drug_feature, pro_feature)
        else:
            combined = torch.cat([drug_feature, pro_feature], dim=1)
            combined = self.fc_concat(combined)

        # 分类器
        x = self.dropout1(self.relu(self.fc1(combined)))  
        x = self.dropout1(self.relu(self.fc2(x)))
        out = self.out(x)
        return out
    
class MMGF_GCN(nn.Module):
    def __init__(self, n_output=1, output_dim=128, num_features_xd=78, 
                num_features_pro=33, num_features_ppi=1442, fingerprint_path=None, fp_dim=2048):
        super().__init__()
        self.output_dim = output_dim
        self.n_output = n_output
        self.fp_dim = fp_dim
        self.pro_combine_fc = nn.Linear(256, 128)# 用于合并后降维

        # 加载分子指纹数据
        if fingerprint_path is not None:
            with open(fingerprint_path, 'rb') as f:
                self.fingerprints = pickle.load(f)
        else:
            self.fingerprints = {}

        # 药物图特征提取（纯GCN架构）
        self.molGconv1 = GCNConv(num_features_xd, num_features_xd*2)
        self.molGconv2 = GCNConv(num_features_xd*2, num_features_xd*4)  
        self.molGconv3 = GCNConv(num_features_xd*4, output_dim)
        self.molGconv4 = GCNConv(output_dim, output_dim)  
        self.molFC1 = nn.Linear(output_dim, 1024)
        self.molFC2 = nn.Linear(1024, output_dim)

        # 蛋白质特征提取（纯GCN架构）
        self.proGconv1 = GCNConv(num_features_pro, output_dim)
        self.proGconv2 = GCNConv(output_dim, output_dim)  
        self.proGconv3 = GCNConv(output_dim, output_dim)
        self.proFC1 = nn.Linear(output_dim, 1024)
        self.proFC2 = nn.Linear(1024, output_dim)

        # PPI特征提取（纯GCN架构）
        self.ppiGconv1 = GCNConv(num_features_ppi, 1024)
        self.ppiGconv2 = GCNConv(1024, 512)  
        self.ppiGconv3 = GCNConv(512, output_dim)
        self.ppiGconv4 = GCNConv(output_dim, output_dim)  

        # 调整全连接层适配新维度
        self.ppiFC1 = nn.Linear(output_dim, 1024)
        self.ppiFC2 = nn.Linear(1024, output_dim)

        # LSTM和分类器（保持不变）
        self.fp_lstm = nn.LSTM(1, output_dim, batch_first=True)
        self.fp_fc = nn.Linear(output_dim, output_dim)
        self.mutual_att = CoAttention(output_dim*2, output_dim, 64)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(384, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, n_output)

    def forward(self, mol_data, pro_data, ppi_edge, ppi_features, pro_graph, mol_smiles=None):
        x, edge_index, batch = mol_data.x, mol_data.edge_index, mol_data.batch
        seq_num = pro_data.seq_num
        p_x, p_edge_index, p_edge_len, p_batch = pro_graph
        # 药物图特征提取
        x = self.relu(self.molGconv1(x, edge_index))
        x = self.relu(self.molGconv2(x, edge_index))
        x = self.relu(self.molGconv3(x, edge_index))
        x = self.relu(self.molGconv4(x, edge_index))
        x = gep(x, batch)  
        x = self.dropout2(self.relu(self.molFC1(x)))
        mol_feature = self.dropout2(self.molFC2(x))  

        # 分子指纹LSTM处理
        if mol_smiles is not None:
            fp_list = [self.fingerprints.get(smile, np.zeros(self.fp_dim)) for smile in mol_smiles]
            fp_tensor = torch.tensor(np.array(fp_list), dtype=torch.float32).to(x.device)
            fp_input = fp_tensor.unsqueeze(-1)  
            
            # LSTM处理
            lstm_out, (hidden, cell) = self.fp_lstm(fp_input)
            fp_feature = self.fp_fc(hidden.squeeze(0))  
            
            # 拼接药物特征
            drug_feature = torch.cat([mol_feature, fp_feature], dim=1) 
        else:
            drug_feature = mol_feature

        # 蛋白质特征提取
        p_x = self.relu(self.proGconv1(p_x, p_edge_index))
        p_x = self.relu(self.proGconv2(p_x, p_edge_index))
        p_x = self.relu(self.proGconv3(p_x, p_edge_index))
        p_x = gep(p_x, p_batch)
        p_x = self.dropout2(self.relu(self.proFC1(p_x)))
        pro_onefeature = self.dropout2(self.proFC2(p_x))  # [379, 128]
        #print(pro_onefeature.shape)
        # PPI特征处理
        #print(ppi_features.shape) [379，1442]
        ppi_edge, _ = dropout_adj(ppi_edge, p=0.6, training=self.training)
        ppi_x = self.relu(self.ppiGconv1(ppi_features, ppi_edge))
        ppi_x = self.relu(self.ppiGconv2(ppi_x, ppi_edge))
        ppi_x = self.relu(self.ppiGconv3(ppi_x, ppi_edge))
        ppi_x = self.relu(self.ppiGconv4(ppi_x, ppi_edge))
        ppi_x = self.dropout1(self.relu(self.ppiFC1(ppi_x)))
        ppi_dualfeature = self.dropout1(self.ppiFC2(ppi_x))[seq_num]  # [256, 128]
        #print(ppi_dualfeature.shape)[512,128]
        # 拼接并降维
        # 从pro_onefeature中选择与当前batch对应的蛋白质特征
        pro_selected = pro_onefeature[seq_num]  # [256, 128]
        pro_feature = torch.cat([pro_selected, ppi_dualfeature], dim=1)  # [256, 256]
        pro_feature = self.pro_combine_fc(pro_feature)  # [256, 128]

        # 特征融合
        combined = self.mutual_att(drug_feature, pro_feature)  # [B, 256+128=384]
        
        # 分类器
        x = self.dropout1(self.relu(self.fc1(combined)))  
        x = self.dropout1(self.relu(self.fc2(x)))
        out = self.out(x)
        return out
    
class CoAttention(nn.Module):
    def __init__(self, drug_dim, protein_dim, k=64):
        super().__init__()
        self.k = k
        
        # 药物到蛋白质的注意力参数
        self.W_q_x = nn.Linear(drug_dim, k)  # 药物查询变换
        self.W_k_p = nn.Linear(protein_dim, k)  # 蛋白质键变换
        self.W_v_p = nn.Linear(protein_dim, k)  # 蛋白质值变换
        
        # 蛋白质到药物的注意力参数
        self.W_q_p = nn.Linear(protein_dim, k)  # 蛋白质查询变换
        self.W_k_x = nn.Linear(drug_dim, k)  # 药物键变换
        self.W_v_x = nn.Linear(drug_dim, k)  # 药物值变换
        
        # 输出变换
        self.fc_x = nn.Linear(k, drug_dim)
        self.fc_p = nn.Linear(k, protein_dim)
        
        # 初始化参数
        nn.init.xavier_normal_(self.W_q_x.weight)
        nn.init.xavier_normal_(self.W_k_p.weight)
        nn.init.xavier_normal_(self.W_v_p.weight)
        nn.init.xavier_normal_(self.W_q_p.weight)
        nn.init.xavier_normal_(self.W_k_x.weight)
        nn.init.xavier_normal_(self.W_v_x.weight)
        nn.init.xavier_normal_(self.fc_x.weight)
        nn.init.xavier_normal_(self.fc_p.weight)

    def forward(self, x, p):
        """
        x: 药物特征 [batch_size, drug_dim]
        p: 蛋白质特征 [batch_size, protein_dim]
        返回融合后的联合特征
        """
        # 药物到蛋白质的注意力
        Q_x = self.W_q_x(x)  # [B, k]
        K_p = self.W_k_p(p)  # [B, k]
        V_p = self.W_v_p(p)  # [B, k]
        
        # 计算注意力得分
        scores_x = torch.bmm(Q_x.unsqueeze(1), K_p.unsqueeze(2))  # [B, 1, 1]
        scores_x = scores_x / (self.k ** 0.5)
        alpha_x = F.softmax(scores_x, dim=-1)
        
        # 蛋白质特征加权
        attended_p = alpha_x * V_p.unsqueeze(1)  # [B, 1, k]
        attended_p = attended_p.squeeze(1)       # [B, k]
        attended_p = self.fc_p(attended_p)       # [B, protein_dim]

        # 蛋白质到药物的注意力
        Q_p = self.W_q_p(p)  # [B, k]
        K_x = self.W_k_x(x)  # [B, k]
        V_x = self.W_v_x(x)  # [B, k]
        
        # 计算注意力得分
        scores_p = torch.bmm(Q_p.unsqueeze(1), K_x.unsqueeze(2))  # [B, 1, 1]
        scores_p = scores_p / (self.k ** 0.5)
        alpha_p = F.softmax(scores_p, dim=-1)
        
        # 药物特征加权
        attended_x = alpha_p * V_x.unsqueeze(1)  # [B, 1, k]
        attended_x = attended_x.squeeze(1)       # [B, k]
        attended_x = self.fc_x(attended_x)       # [B, drug_dim]

        # 拼接融合特征
        fused = torch.cat([attended_x, attended_p], dim=1)  # [B, drug_dim+protein_dim]
        return fused



class MMGF_GraphSage(nn.Module):
    def __init__(self, n_output=1, output_dim=128, num_features_xd=78, 
                 num_features_pro=33, num_features_ppi=1442,fingerprint_path=None, fp_dim=2048):
        super().__init__()
        self.output_dim = output_dim
        self.n_output = n_output
        self.fp_dim = fp_dim
        self.pro_combine_fc = nn.Linear(256, 128)  # 用于合并后降维

        # 加载分子指纹数据
        if fingerprint_path is not None:
            with open(fingerprint_path, 'rb') as f:
                self.fingerprints = pickle.load(f)
        else:
            self.fingerprints = {}

        # 药物图特征提取
        self.molGconv1 = GraphSAGE(in_channels=num_features_xd, hidden_channels=num_features_xd*2, num_layers=1, aggr='mean')
        self.molGconv2 = GraphSAGE(in_channels=num_features_xd*2, hidden_channels=num_features_xd*4, num_layers=1, aggr='mean')
        self.molGconv3 = GraphSAGE(in_channels=num_features_xd*4, hidden_channels=output_dim, num_layers=1, aggr='mean')
        self.molFC1 = nn.Linear(output_dim, 1024)
        self.molFC2 = nn.Linear(1024, output_dim)

        # 蛋白质特征提取
        self.proGconv1 = GraphSAGE(in_channels=num_features_pro, hidden_channels=output_dim, num_layers=1, aggr='mean')
        self.proGconv2 = GraphSAGE(in_channels=output_dim, hidden_channels=output_dim, num_layers=1, aggr='mean')
        self.proGconv3 = GraphSAGE(in_channels=output_dim, hidden_channels=output_dim, num_layers=1, aggr='mean')
        self.proFC1 = nn.Linear(output_dim, 1024)
        self.proFC2 = nn.Linear(1024, output_dim)

        # PPI特征提取
        self.ppiGconv1 = GraphSAGE(
            in_channels=num_features_ppi,  # 改为PPI原始特征维度
            hidden_channels=1024, 
            num_layers=1, 
            aggr='mean'
        )
        self.ppiGconv2 = GraphSAGE(in_channels=1024, hidden_channels=output_dim, num_layers=1, aggr='mean')
        self.ppiFC1 = nn.Linear(output_dim, 1024)
        self.ppiFC2 = nn.Linear(1024, output_dim)

        # LSTM
        self.fp_lstm = nn.LSTM(
            input_size=1,          # 每个指纹位视为1维特征
            hidden_size=output_dim, 
            batch_first=True,
            bidirectional=False
        )
        self.fp_fc = nn.Linear(output_dim, output_dim)  

        # 修改注意力模块初始化
        self.mutual_att = CoAttention(
            drug_dim=output_dim * 2,  # 保持256维度
            protein_dim=output_dim,    # 保持128维度
            k=64
        )

        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)

        # 分类器输入维度调整
        self.fc1 = nn.Linear(256 + 128, 1024)  # 256+128=384
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)


    def forward(self, mol_data, pro_data, ppi_edge, ppi_features, pro_graph, mol_smiles=None):
        x, edge_index, batch = mol_data.x, mol_data.edge_index, mol_data.batch
        seq_num = pro_data.seq_num
        p_x, p_edge_index, p_edge_len, p_batch = pro_graph
        # 药物图特征提取
        x = self.relu(self.molGconv1(x, edge_index))
        x = self.relu(self.molGconv2(x, edge_index))
        x = self.relu(self.molGconv3(x, edge_index))
        x = gep(x, batch)  
        x = self.dropout2(self.relu(self.molFC1(x)))
        mol_feature = self.dropout2(self.molFC2(x))  

        # 分子指纹LSTM处理
        if mol_smiles is not None:
            fp_list = [self.fingerprints.get(smile, np.zeros(self.fp_dim)) for smile in mol_smiles]
            fp_tensor = torch.tensor(np.array(fp_list), dtype=torch.float32).to(x.device)
            fp_input = fp_tensor.unsqueeze(-1)  
            
            # LSTM处理
            lstm_out, (hidden, cell) = self.fp_lstm(fp_input)
            fp_feature = self.fp_fc(hidden.squeeze(0))  
            
            # 拼接药物特征
            drug_feature = torch.cat([mol_feature, fp_feature], dim=1) 
        else:
            drug_feature = mol_feature

        # 蛋白质特征提取
        p_x = self.relu(self.proGconv1(p_x, p_edge_index))
        p_x = self.relu(self.proGconv2(p_x, p_edge_index))
        p_x = self.relu(self.proGconv3(p_x, p_edge_index))
        p_x = gep(p_x, p_batch)
        p_x = self.dropout2(self.relu(self.proFC1(p_x)))
        pro_onefeature = self.dropout2(self.proFC2(p_x))  # [379, 128]
        #print(pro_onefeature.shape)
        # PPI特征处理
        #print(ppi_features.shape) [379，1442]
        ppi_edge, _ = dropout_adj(ppi_edge, p=0.6, training=self.training)
        ppi_x = self.relu(self.ppiGconv1(ppi_features, ppi_edge))
        ppi_x = self.relu(self.ppiGconv2(ppi_x, ppi_edge))
        ppi_x = self.dropout1(self.relu(self.ppiFC1(ppi_x)))
        ppi_dualfeature = self.dropout1(self.ppiFC2(ppi_x))[seq_num]  # [256, 128]
        #print(ppi_dualfeature.shape)[512,128]
        # 拼接并降维
        # 从pro_onefeature中选择与当前batch对应的蛋白质特征
        pro_selected = pro_onefeature[seq_num]  # [256, 128]
        pro_feature = torch.cat([pro_selected, ppi_dualfeature], dim=1)  # [256, 256]
        pro_feature = self.pro_combine_fc(pro_feature)  # [256, 128]

        # 特征融合
        combined = self.mutual_att(drug_feature, pro_feature)  # [B, 256+128=384]
        
        # 分类器
        x = self.dropout1(self.relu(self.fc1(combined)))  
        x = self.dropout1(self.relu(self.fc2(x)))
        out = self.out(x)
        return out
    
class AblationFactory:
    """消融模型工厂类，根据参数创建不同的消融模型"""
    @staticmethod
    def create_model(model_type, ablation_type=None, **kwargs):
        if model_type == "gcn":
            return MMGF_GCN(**kwargs)
        elif model_type == "graphsage":
            return MMGF_GraphSage(**kwargs)
        else:
            return AblationMMGF(ablation=ablation_type, **kwargs)
