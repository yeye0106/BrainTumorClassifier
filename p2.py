import torch
import torch.nn as nn
from tqdm import tqdm
import config
from dataset import get_data_loaders
from model import get_model
import os


def evaluate_with_tta():
    print(f"[*] 启动全量评估 | 设备: {config.DEVICE} (已开启 TTA 测试时增强)")

    # 获取数据加载器
    _, test_loader, classes = get_data_loaders()

    # 加载模型
    model = get_model(config.NUM_CLASSES).to(config.DEVICE)
    model_path = 'best_brain_tumor_model.pth'

    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE, weights_only=True))
    model.eval()  # 依然保持评估模式

    criterion = nn.CrossEntropyLoss()

    test_loss = 0.0
    correct = 0
    total = 0
    class_correct = [0] * config.NUM_CLASSES
    class_total = [0] * config.NUM_CLASSES

    print(f"[*] 正在对 {len(test_loader.dataset)} 张测试图片进行 TTA 综合预测...")

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating with TTA"):
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)

            # --- TTA 核心逻辑开始 ---
            # 视角 1：原图预测
            outputs_orig = model(images)

            # 视角 2：水平翻转预测 (医学影像的左右脑对称性)
            images_flipped = torch.flip(images, dims=[3])
            outputs_flipped = model(images_flipped)

            # 综合视角：将两次的预测结果求平均
            outputs_final = (outputs_orig + outputs_flipped) / 2.0
            # --- TTA 核心逻辑结束 ---

            # 计算 Loss (仅供参考，以原图 Loss 为主)
            loss = criterion(outputs_orig, labels)
            test_loss += loss.item() * images.size(0)

            # 根据综合视角的预测结果得出最终结论
            _, predicted = torch.max(outputs_final, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for i in range(len(labels)):
                label = labels[i]
                pred = predicted[i]
                if label == pred:
                    class_correct[label] += 1
                class_total[label] += 1

    avg_loss = test_loss / total
    overall_acc = 100 * correct / total

    print("\n" + "=" * 35)
    print(f"🏆 TTA 增强测试完成！总体结果:")
    print(f"平均损失 (Loss): {avg_loss:.4f}")
    print(f"总体准确率 (Accuracy): {overall_acc:.2f}%")
    print("=" * 35)

    print("\n各类别详细表现:")
    for i in range(config.NUM_CLASSES):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            print(f" - {classes[i]:<12}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})")
        else:
            print(f" - {classes[i]:<12}: 无测试样本")


if __name__ == '__main__':
    evaluate_with_tta()