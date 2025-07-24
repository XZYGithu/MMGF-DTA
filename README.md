# MMGF-DTA training 
python train_MMGF_DTA.py

# Run pure GCN architecture experiment 
python ablation_study.py --model_type gcn --dataset davis

# Run pure GraphSAGE architecture experiments 
python ablation_study.py --model_type graphsage --dataset kiba

# Run PPI removal experiment for hybrid architecture 
python ablation_study.py --model_type default --ablation no_ppi --dataset davis

# Run fingerprint removal experiments for hybrid architectures 
python ablation_study.py --model_type default --ablation no_fp --dataset kiba