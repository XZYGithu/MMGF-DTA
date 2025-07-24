# -*- coding:utf-8 -*-
import json
from rdkit import Chem
import numpy as np
import networkx as nx
import pickle
import pandas as pd
from collections import OrderedDict
import argparse

# 构建原子特征的函数
def atom_features(atom):
    #one_of_k_encoding_unk独热编码方法，将原子的化学符号转化为二进制向量
    #atom.GetSymbol() 返回原子的化学符号（例如 'C' 表示碳，'N' 表示氮等）。
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    #one_of_k_encoding独热编码方法，将原子的度数转化为二进制向量。
                    #atom.GetDegree() 返回原子的度数（即与该原子直接相连的其他原子的数量）。
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +         
                    #atom.GetTotalNumHs() 返回与该原子直接相连的氢原子数量。
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    #atom.GetImplicitValence() 返回原子的隐式价电子数（即未明确表示的价电子数）
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                     #返回一个布尔值，表示该原子是否属于芳香环。
                    [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
    #使用 RDKit 的 Chem.MolFromSmiles 函数将 SMILES 字符串转换为 RDKit 的分子对象 mol
    mol = Chem.MolFromSmiles(smile)
    #c_size 药物分子中原子的数量。
    c_size = mol.GetNumAtoms()

    #features节点特征矩阵，每个原子的特征向量。
    features = []
    for atom in mol.GetAtoms():

        feature = atom_features(atom)                       #遍历分子中的每个原子，调用 atom_features(atom) 函数获取原子的特征向量。
        features.append(feature / sum(feature))          #对特征向量进行归一化（feature / sum(feature)），并将其添加到 features 列表中。
                                                        #features 是一个二维列表，每一行表示一个原子的特征向量


    edges = []
    for bond in mol.GetBonds():             #遍历分子中的每个化学键，获取键的起始原子索引和结束原子索引。将这些索引对存储在 edges 列表中。
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()       #使用 networkx 库将 edges 转换为一个无向图 g，然后将其转换为有向图（to_directed()）
    edge_index = []
    #遍历图中的所有边，将边的起始和结束原子索引存储在 edge_index 列表中。 edge_index 是一个二维列表，每一行表示一条边的起始和结束原子索引。
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
    assert len(edge_index) != 0     #确保 edge_index 不为空（即分子中至少有一条边）。

    return c_size, features, edge_index


#这段代码的主要功能是对蛋白质残基（氨基酸）的多种特征进行归一化处理，并将这些特征存储在不同的字典中。
def dic_normalize(dic):
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0
    return dic

# 残基类型表
pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y','X']
# 脂肪族残基
pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
# 芳香族残基
pro_res_aromatic_table = ['F', 'W', 'Y']
# 极性中性残基
pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
# 酸性带电残基
pro_res_acidic_charged_table = ['D', 'E']
# 碱性带电残基
pro_res_basic_charged_table = ['H', 'K', 'R']


#残基特征表
# 残基的分子量
res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                    'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                    'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}

#残基的羧基解离常数
res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}
#残基的氨基解离常数
res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
                 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
                 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}
#残基的其他基团解离常数
res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
                 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}
#残基的等电点（pI）。
res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
                'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
                'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}

#残基在 pH=2 时的疏水性。
res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100,
                             'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7,
                             'T': 13, 'V': 79, 'W': 84, 'Y': 49}

#：残基在 pH=7 时的疏水性。
res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                             'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                             'T': 13, 'V': 76, 'W': 97, 'Y': 63}

#归一化处理
#对每个特征表调用 dic_normalize 函数，将其值归一化到 [0, 1] 范围内。
#归一化后的特征表可以用于机器学习模型，确保不同特征具有相同的尺度。
res_weight_table = dic_normalize(res_weight_table)
res_pka_table = dic_normalize(res_pka_table)
res_pkb_table = dic_normalize(res_pkb_table)
res_pkx_table = dic_normalize(res_pkx_table)
res_pl_table = dic_normalize(res_pl_table)
res_hydrophobic_ph2_table = dic_normalize(res_hydrophobic_ph2_table)
res_hydrophobic_ph7_table = dic_normalize(res_hydrophobic_ph7_table)



#这段代码定义了一个函数 residue_features(residue)，用于生成蛋白质残基（氨基酸）的特征向量。特征向量包括残基的类别属性（如脂肪族、芳香族等）以及物理化学属性（如分子量、pKa、疏水性等）。
def residue_features(residue):
    # 检查残基是否合法.如果输入的残基不在预定义的残基表 pro_res_table 中，则将其替换为 'X'（表示未知或非标准残基）。
    if residue not in pro_res_table:
        residue = 'X'
    #生成残基的类别属性.res_property1 是一个包含 5 个二进制值的列表，表示残基的类别属性.如果残基属于某一类别，则对应位置为 1，否则为 0
    res_property1 = [
    1 if residue in pro_res_aliphatic_table else 0,  # 是否为脂肪族残基
    1 if residue in pro_res_aromatic_table else 0,   # 是否为芳香族残基
    1 if residue in pro_res_polar_neutral_table else 0,  # 是否为极性中性残基
    1 if residue in pro_res_acidic_charged_table else 0,  # 是否为酸性带电残基
    1 if residue in pro_res_basic_charged_table else 0  # 是否为碱性带电残基
]
    #生成残基的物理化学属性
    #    res_property2 是一个包含 7 个浮点数的列表，表示残基的物理化学属性：
    res_property2 = [
    res_weight_table[residue],  # 分子量
    res_pka_table[residue],    # 羧基解离常数 (pKa)
    res_pkb_table[residue],    # 氨基解离常数 (pKb)
    res_pkx_table[residue],    # 其他基团解离常数 (pKx)
    res_pl_table[residue],     # 等电点 (pI)
    res_hydrophobic_ph2_table[residue],  # pH=2 时的疏水性
    res_hydrophobic_ph7_table[residue]   # pH=7 时的疏水性
]
    #这个函数的作用是将蛋白质残基（氨基酸）转换为一个特征向量，包含类别属性和物理化学属性
    return np.array(res_property1 + res_property2)




#用于生成蛋白质序列中每个残基的特征描述符。特征描述符包括两个部分：
    #独热编码（one-hot encoding）：表示残基的类型。
    #残基特征：包括类别属性和物理化学属性。
#最终，函数返回一个二维数组，每一行表示一个残基的特征向量。
def seq_feature(pro_seq): 
    #初始化特征矩阵
    #pro_hot：用于存储每个残基的独热编码，形状为 (len(pro_seq), len(pro_res_table))。
    #len(pro_seq) 是蛋白质序列的长度。
    #len(pro_res_table) 是残基类型的总数（包括 'X'）。
    pro_hot = np.zeros((len(pro_seq), len(pro_res_table)))
    #pro_property：用于存储每个残基的特征向量，形状为 (len(pro_seq), 12)。 每个残基的特征向量长度为 12（由 residue_features 函数生成）。
    pro_property = np.zeros((len(pro_seq), 12))

    #   对蛋白质序列中的每个残基：

        #使用 one_of_k_encoding_unk 函数生成残基的独热编码，并存储在 pro_hot[i,] 中。

        #使用 residue_features 函数生成残基的特征向量，并存储在 pro_property[i,] 中。
    for i in range(len(pro_seq)):
        pro_hot[i,] = one_of_k_encoding_unk(pro_seq[i], pro_res_table)
        pro_property[i,] = residue_features(pro_seq[i])
        #将每个残基的独热编码和特征向量进行拼接，生成最后的特征矩阵
        #每个残基的特征向量包括：21 维独热编码：表示残基的类型。 12 维残基特征：包括类别属性和物理化学属性。
    return np.concatenate((pro_hot, pro_property), axis=1)




#将蛋白质序列转化为图结构。包括（节点特征矩阵；邻接矩阵（通过接触图表示）；节点数量（蛋白质序列的长度）。）
def seq_to_graph(pro_id, seq, db):
    #输入：
    #ro_id：蛋白质的 UniProt ID（唯一标识符）。
    #seq：蛋白质序列（字符串形式）。
    #db：数据集名称
    sparse_edge_matrix = np.load(f'data/{db}/contact_map/{pro_id}.npy') # 加载接触图
    edge_index = np.argwhere(sparse_edge_matrix == 1).transpose(1, 0)#提取边的索引

    c_size = len(seq)# 计算节点数量
    features = seq_feature(seq) #生成蛋白质序列中每个残基的特征矩阵形状：（蛋白质序列长度，33）。 33 是每个残基的特征向量长度（21 维独热编码 + 12 维残基特征）。
    return c_size, features, edge_index



#训练集和测试集是按照 4：1 的比例划分
def data_split(dataset):
    """
    将数据集划分为训练集和测试集，并将数据转换为 CSV 文件。
    :param dataset: 数据集名称（如 'davis' 或 'kiba'）
    """
    print('正在处理数据集: ', dataset)
    
    # 数据集路径
    fpath = 'data/' + dataset + '/'
    
    # 加载训练集和测试集的索引文件
    train_fold = json.load(open(fpath + "folds/train_fold_setting1.txt"))   
    train_fold = [ee for e in train_fold for ee in e ]  # 将嵌套列表展平
    valid_fold = json.load(open(fpath + "folds/test_fold_setting1.txt"))
    
    # 加载药物分子 SMILES 和蛋白质序列
    ligands = json.load(open(fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)
    proteins = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)
    
    # 加载药物-蛋白质对的亲和力值
    affinity = pickle.load(open(fpath + "Y","rb"), encoding='latin1')
    
    # 初始化药物分子和蛋白质序列的列表
    drugs = []
    prots = []
    
    # 遍历药物分子 SMILES，转换为标准格式并存入列表
    for d in ligands.keys():
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
        drugs.append(lg)
    
    # 遍历蛋白质序列，存入列表
    for t in proteins.keys():
        prots.append(proteins[t])
    
    # 如果数据集是 Davis，对亲和力值进行负对数变换
    if dataset == 'davis':
        affinity = [-np.log10(y/1e9) for y in affinity]  # 将亲和力值转换为 pKd 或 pKi 形式
    
    # 将亲和力值转换为 NumPy 数组
    affinity = np.asarray(affinity)
    
    # 定义训练集和测试集的处理选项
    opts = ['train', 'test']
    for opt in opts:
        # 找到亲和力值不为 NaN 的行和列索引
        rows, cols = np.where(np.isnan(affinity) == False)
        
        # 根据当前选项（训练集或测试集）筛选数据
        if opt == 'train':
            rows, cols = rows[train_fold], cols[train_fold]  # 使用训练集索引
        elif opt == 'test':
            rows, cols = rows[valid_fold], cols[valid_fold]  # 使用测试集索引
        
        # 将筛选后的数据写入 CSV 文件
        with open('data/' + dataset + '/' + opt + '.csv', 'w') as f:
            f.write('compound_iso_smiles,target_sequence,affinity\n')  # 写入 CSV 文件头
            for pair_ind in range(len(rows)):
                ls = []  # 初始化当前行的数据列表
                ls += [ drugs[rows[pair_ind]]  ]  # 添加药物分子 SMILES
                ls += [ prots[cols[pair_ind]]  ]  # 添加蛋白质序列
                ls += [ affinity[rows[pair_ind],cols[pair_ind]]  ]  # 添加亲和力值
                f.write(','.join(map(str, ls)) + '\n')  # 将列表转换为字符串并写入文件
    
    # 打印数据集信息
    print('\n数据集:', dataset)
    print('训练集大小:', len(train_fold))
    print('测试集大小:', len(valid_fold))
    print('唯一药物分子数量, 唯一蛋白质数量:', len(set(drugs)), len(set(prots)))


#构建药物分子（SMILES）和蛋白质序列的图结构 mol_data.pkl pro_data.pkl
def construct_graph(args):
    print('Construct graph for ', args.dataset)
    ## 1. generate drug graph dict.
    compound_iso_smiles = []
    if args.dataset == 'Human':
        opts = ['train1', 'test1']
    else:
        opts = ['train', 'test']
    for opt in opts:
        df = pd.read_csv(f'data/{args.dataset}/' + opt + '.csv')
        compound_iso_smiles += list(df['compound_iso_smiles'])
    compound_iso_smiles = set(compound_iso_smiles)
    smile_graph = {}
    for smile in compound_iso_smiles:
        g = smile_to_graph(smile)
        smile_graph[smile] = g
    print('drug graph is constructed successfully!')

    ## 2.generate protein graph dict.
    seq_dict = {}
    with open(f'data/{args.dataset}/{args.dataset}_dict.txt', 'r') as file:
        for line in file.readlines():
            line = line.lstrip('>').strip().split('\t')
            seq_dict[line[1]] = line[0]
    seq_graph = {}
    for pro_id, seq in seq_dict.items():
        g = seq_to_graph(pro_id, seq, args.dataset)
        seq_graph[seq] = g
    print('protein graph is constructed successfully!')

    ## 3. Serialized graph data
    with open(f'data/{args.dataset}/mol_data.pkl', 'wb') as smile_file:
        pickle.dump(smile_graph, smile_file)
    with open(f'data/{args.dataset}/pro_data.pkl', 'wb') as seq_file:
        pickle.dump(seq_graph, seq_file)



#这个函数的作用是将 Davis 数据集的训练集划分为 5 个子集，支持 5 折交叉验证。

#每个子集的测试集占 20%，训练集占 80%。
def fold_split_for_davis():
    df = pd.read_csv('data/davis/train.csv')
    portion = int(0.2 * len(df['affinity']))
    for fold in range(5):
        if fold < 4:
            df_test = df.iloc[fold * portion:(fold + 1) * portion]
            df_train = pd.concat([df.iloc[:fold * portion], df.iloc[(fold + 1) * portion:]], ignore_index=True)
        if fold == 4:
            df_test = df.iloc[fold * portion:]
            df_train = df.iloc[:fold * portion]
        assert (len(df_test) + len(df_train)) == len(df)
        df_test.to_csv(f'data/davis/5 fold/test{fold + 1}.csv', index=False)
        df_train.to_csv(f'data/davis/5 fold/train{fold + 1}.csv', index=False)


def main(args):
    data_split(args.dataset)
    construct_graph(args)
    fold_split_for_davis()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default= 'davis', help='dataset name',choices=['davis','kiba'])
    args = parser.parse_args()
    main(args)
