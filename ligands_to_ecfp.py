import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
import json
import numpy as np
from rdkit import DataStructs

# 读取ligands_can.txt文件
ligand_file_path = '/home/xuzhiyuan/HiSIF-DTA-main/data/davis/ligands_can.txt'

with open(ligand_file_path, 'r') as f:
    ligands_data = json.load(f)

# 提取摩根指纹的函数
def extract_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        # 提取ECFP4分子指纹
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        # 将ExplicitBitVect对象转换为NumPy数组
        fp_array = np.zeros((1,), dtype=int)
        DataStructs.ConvertToNumpyArray(fp, fp_array)
        return fp_array
    else:
        return None

# 创建字典来存储SMILES和对应的摩根指纹
fingerprints = {}

# 提取每个药物的摩根指纹
for key, smiles in ligands_data.items():
    fingerprint = extract_fingerprint(smiles)
    if fingerprint is not None:
        fingerprints[smiles] = fingerprint

# 保存到PKL文件
output_pkl_path = '/home/xuzhiyuan/HiSIF-DTA-main/ligand_to_ecfp.pkl'
with open(output_pkl_path, 'wb') as pkl_file:
    pickle.dump(fingerprints, pkl_file)

print(f"摩根指纹已保存到 {output_pkl_path}")
