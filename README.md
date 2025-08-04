# 运行两阶段训练

python two_stage_train.py --data_path /path/to/arc/data --save_dir ./checkpoints/two_stage --gpu_id 0 --ae_epochs 50 --rule_epochs 150 --finetune_ae

# 如果想从现有的自编码器预训练模型开始

python two_stage_train.py --data_path /path/to/arc/data --save_dir ./checkpoints/two_stage --resume_ae_path ./checkpoints/pretrain_existing/best_autoencoder.pt --rule_epochs 200

# 如果想继续训练中断的规则训练

python two_stage_train.py --data_path /path/to/arc/data --save_dir ./checkpoints/two_stage --resume_ae_path ./checkpoints/pretrain_existing/best_autoencoder.pt --resume_rule_path ./checkpoints/rule_train_existing/rule_guided_vae_epoch_50.pt
