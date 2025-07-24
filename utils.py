
from torch_geometric.data import Batch,InMemoryDataset
from torch_geometric.data.dataset import Dataset, IndexType
from torch_geometric import data as DATA
import torch
import pickle
from torch_geometric.loader import DataLoader
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import numpy as np

def collate(data):
    drug_batch = Batch.from_data_list( [item[0] for item in data])
    seq_batch = Batch.from_data_list([item[1] for item in data])
    return drug_batch,seq_batch

def collate2(data):
    seq_batch = Batch.from_data_list([item for item in data])
    return seq_batch


def get_keys(d, value):
    for k, v in d.items():
        if v == value:
            return k


class DTADataset(Dataset):
    def __init__(self, smile_list, seq_list, label_list, mol_data=None, ppi_index=None, smiles=None):
        super(DTADataset, self).__init__()
        self.smile_list = smile_list
        self.seq_list = seq_list
        self.label_list = label_list
        self.smile_graph = mol_data
        self.ppi_index = ppi_index
        self.smiles = smiles  # 新增的字段，用于存储smiles

    def __len__(self):
        return len(self.smile_list)

    def __getitem__(self, index):
        smile = self.smile_list[index]
        seq = self.seq_list[index]
        labels = self.label_list[index]

        # 获取药物的图数据
        drug_size, drug_features, drug_edge_index = self.smile_graph[smile]
        seq_size = len(seq)
        seq_index = self.ppi_index[seq]

        # Wrapping graph data into the Data format supported by PyG (PyTorch Geometric).
        Data_smile = DATA.Data(x=torch.Tensor(drug_features), edge_index=torch.LongTensor(drug_edge_index).transpose(1, 0), y=torch.FloatTensor([labels]))
        Data_smile.__setitem__('c_size', torch.LongTensor([drug_size]))
        
        # 这里返回SMILES，作为模型的输入
        Data_seq = DATA.Data(y=torch.FloatTensor([labels]), seq_num=torch.LongTensor([seq_index]))  # The seq_index indicates the node number of the protein in the PPI graph.
        Data_seq.__setitem__('c_size', torch.LongTensor([seq_size]))
        
        # 返回SMILES，供模型使用
        return Data_smile, Data_seq, self.smiles[index]  # 返回smiles作为第三项


class GraphDataset(InMemoryDataset):
    def __init__(self, root='/tmp', graph = None,index = None ,type=None):
        super(GraphDataset, self).__init__(root)
        self.type = type
        self.index = index
        self.process(graph,index)

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        pass

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        pass

    def process(self, graph,index):
        data_list = []
        count = 0
        for key in index:
            size, features, edge_index = graph[key]
            # Wrapping graph data into the Data format supported by PyG (PyTorch Geometric).
            Data = DATA.Data(x=torch.Tensor(features), edge_index=torch.LongTensor(edge_index),graph_num = torch.LongTensor([count]))
            Data.__setitem__('c_size', torch.LongTensor([size]))
            count += 1
            data_list.append(Data)
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def proGraph(graph_data, index, device):
    proGraph_dataset = GraphDataset(graph=graph_data, index=index ,type = 'pro')
    proGraph_loader = DataLoader(proGraph_dataset, batch_size=len(graph_data), shuffle=False)
    pro_graph = None
    for batchid, batch in enumerate(proGraph_loader):
        pro_graph = batch.x.to(device),batch.edge_index.to(device),batch.graph_num.to(device),batch.batch.to(device)
    return pro_graph

def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse

def mse_print(y,f):
    mse = ((y - f)**2)
    return mse

def calculate_rm2(y_true, y_pred):
    # 计算有截距的r²
    r2 = r2_score(y_true, y_pred)
    
    # 计算无截距的r²
    model = LinearRegression(fit_intercept=False)
    model.fit(y_pred.reshape(-1,1), y_true)
    y_pred_no_intercept = model.predict(y_pred.reshape(-1,1))
    r0_2 = r2_score(y_true, y_pred_no_intercept)
    
    # 处理可能的负数情况
    adjusted_term = max(0, r2 - r0_2)
    rm2 = r2 * (1 - np.sqrt(adjusted_term))
    return rm2

def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-10)
