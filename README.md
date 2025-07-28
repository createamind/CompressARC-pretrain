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
