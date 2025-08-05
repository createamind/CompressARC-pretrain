import os
import argparse
import torch
from pretrain_autoencoder import pretrain_autoencoder
from train_rule_guided import train_rule_guided_from_pretrained

def two_stage_training(
    data_path,
    save_dir,
    ae_epochs=100,
    rule_epochs=200,
    ae_batch_size=16,
    rule_batch_size=4,
    ae_lr=1e-3,
    rule_lr=5e-4,
    recon_weight=5.0,
    vq_weight=0.1,
    rule_weight=1.0,
    val_split=0.1,
    gpu_id=None,
    resume_ae_path="",
    resume_rule_path="",
    finetune_ae=True,
    seed=42
):
    """执行两阶段训练过程"""
    print("=" * 50)
    print("阶段 1: 编码解码器预训练")
    print("=" * 50)

    passaetrain = False

    # 预训练自编码器
    if resume_ae_path and passaetrain:
        print(f"跳过自编码器预训练，使用现有权重: {resume_ae_path}")
        autoencoder_path = resume_ae_path
    else:
        autoencoder_path = pretrain_autoencoder(
            data_path=data_path,
            save_dir=save_dir,
            epochs=ae_epochs,
            batch_size=ae_batch_size,
            learning_rate=ae_lr,
            gpu_id=gpu_id,
            resume_path=resume_ae_path,
            seed=seed
        )

    print("\n" + "=" * 50)
    print("阶段 2: 规则提取和应用训练")
    print("=" * 50)

    # 训练规则提取和应用
    model, run_dir = train_rule_guided_from_pretrained(
        pretrained_path=autoencoder_path,
        data_path=data_path,
        save_dir=save_dir,
        epochs=rule_epochs,
        batch_size=rule_batch_size,
        learning_rate=rule_lr,
        recon_weight=recon_weight,
        vq_weight=vq_weight,
        rule_weight=rule_weight,
        gpu_id=gpu_id,
        resume_path=resume_rule_path,
        val_split=val_split,
        finetune_ae=finetune_ae,
        seed=seed
    )

    print("\n" + "=" * 50)
    print(f"训练完成! 最终模型保存于: {os.path.join(run_dir, 'best_model.pt')}")
    print("=" * 50)

    return model, run_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARC任务的两阶段训练")

    # parser.add_argument("--data_path", type=str, required=True, help="数据集路径")
    parser.add_argument("--data_path", type=str, default="data/training",
                        help="Path to ARC training data")
    parser.add_argument("--save_dir", type=str, default="./checkpoints/two_stage", help="保存目录")

    # 自编码器预训练参数
    parser.add_argument("--ae_epochs", type=int, default=100, help="自编码器预训练轮次")
    parser.add_argument("--ae_batch_size", type=int, default=16, help="自编码器预训练批量大小")
    parser.add_argument("--ae_lr", type=float, default=1e-3, help="自编码器预训练学习率")
    parser.add_argument("--resume_ae_path", type=str, default="", help="恢复自编码器预训练路径")

    # 规则训练参数
    parser.add_argument("--rule_epochs", type=int, default=200, help="规则训练轮次")
    parser.add_argument("--rule_batch_size", type=int, default=4, help="规则训练批量大小")
    parser.add_argument("--rule_lr", type=float, default=5e-4, help="规则训练学习率")
    parser.add_argument("--recon_weight", type=float, default=5.0, help="重建损失权重")
    parser.add_argument("--vq_weight", type=float, default=0.1, help="VQ损失权重")
    parser.add_argument("--rule_weight", type=float, default=1.0, help="规则损失权重")
    parser.add_argument("--val_split", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--resume_rule_path", type=str, default="", help="恢复规则训练路径")

    # 其他参数
    parser.add_argument("--gpu_id", type=int, default=None, help="GPU ID")
    parser.add_argument("--finetune_ae", action="store_true", help="是否微调自编码器")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 执行两阶段训练
    model, run_dir = two_stage_training(
        data_path=args.data_path,
        save_dir=args.save_dir,
        ae_epochs=args.ae_epochs,
        rule_epochs=args.rule_epochs,
        ae_batch_size=args.ae_batch_size,
        rule_batch_size=args.rule_batch_size,
        ae_lr=args.ae_lr,
        rule_lr=args.rule_lr,
        recon_weight=args.recon_weight,
        vq_weight=args.vq_weight,
        rule_weight=args.rule_weight,
        val_split=args.val_split,
        gpu_id=args.gpu_id,
        resume_ae_path=args.resume_ae_path,
        resume_rule_path=args.resume_rule_path,
        finetune_ae=args.finetune_ae,
        seed=args.seed
    )

    print(f"训练结果保存在: {run_dir}")