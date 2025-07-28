README.md

python test_time_train.py --model checkpoints/final_model.pt --data path/to/test/data --tt_epochs 10 --steps 5

python train.py --data path/to/arc/data --save_dir checkpoints

python train.py --data /path/to/data --save_dir ./checkpoints/ --epochs 100 --batch_size 32 --lr 0.001 --accumulation_steps 4

python test.py --model ./checkpoints/final_model.pt --data /path/to/test/data --steps 5 --tt_epochs 10
