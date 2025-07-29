
使用说明

1. 训练层次化VAE模型
bash
python train.py --data /path/to/data --hierarchical_vae --recon_weight 2.0 --vq_weight 0.5
这将使用层次化面向对象VAE模型，并将重建损失权重设为2.0，VQ损失权重设为0.5，以解决之前观察到的VQ损失占主导地位的问题。

2. 测试层次化VAE模型
bash
python test.py --model /path/to/model.pt --data /path/to/test/data --hierarchical_vae
3. 运行参数扫描
bash
python test.py --model /path/to/model.pt --data /path/to/test/data --hierarchical_vae --param_scan
架构优势
对象连通性分析：自动提取ARC任务中的连通对象，将像素集合视为完整对象
层次化表示：同时学习像素级模式、对象级结构和关系级互动
多路径解码：结合不同层级的特征进行解码，确保低级细节和高级概念都被保留
平衡损失权重：通过可调整的权重解决VQ损失过大的问题

before 7.29

README.md

python test_time_train.py --model checkpoints/final_model.pt --data path/to/test/data --tt_epochs 10 --steps 5

python train.py --data path/to/arc/data --save_dir checkpoints

python train.py --data /path/to/data --save_dir ./checkpoints/ --epochs 100 --batch_size 32 --lr 0.001 --accumulation_steps 4

python test.py --model ./checkpoints/final_model.pt --data /path/to/test/data --steps 5 --tt_epochs 10

离散化：
使用方法
现在您可以使用以下命令来训练模型：

使用默认的离散VAE训练：
python train.py --data /path/to/data --save_dir ./discrete_checkpoints/

使用标准VAE训练：
python train.py --data /path/to/data --save_dir ./standard_checkpoints/ --standard_vae

自定义离散VAE参数：
python train.py --data /path/to/data --save_dir ./custom_discrete/ --codebook_size 1024 --embedding_dim 128

测试离散VAE模型：
python test.py --model ./discrete_checkpoints/final_discrete_model

参数扫描
python test.py --model ./checkpoints/1234567890/final_discrete_model.pt --data /path/to/test/data --param_scan

gpu：
训练模型（指定GPU）：
bash

# 使用GPU 0训练

python train.py --data /path/to/data --save_dir ./checkpoints/ --gpu 0

# 使用GPU 1训练

python train.py --data /path/to/data --save_dir ./checkpoints/ --gpu 1
测试模型（指定GPU）：
bash

# 在GPU 0上测试

python test.py --model ./checkpoints/20230728_100125/final_discrete_model.pt --data /path/to/test/data --gpu 0

# 参数扫描

python test.py --model ./checkpoints/20230728_100125/final_discrete_model.pt --data /path/to/test/data --param_scan --gpu 0
