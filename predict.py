import sys
import torch
import torch.nn as nn
from tqdm import tqdm
from config import *
from dataset import get_test_dataloader
from model import get_model


def predict():
    test_loader, classes = get_test_dataloader()

    model = get_model()
    try:
        model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE, weights_only=True))
    except FileNotFoundError:
        print("未找到 best_model.pth，请先运行 train.py 进行训练！", flush=True)
        return

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(DEVICE)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    test_loss = 0.0
    test_correct = 0
    test_total = 0

    class_correct = {classname: 0 for classname in classes}
    class_total = {classname: 0 for classname in classes}

    print("正在使用 Testing 数据集进行全量标准预测...", flush=True)
    with torch.no_grad():
        with tqdm(test_loader, desc="Predicting", file=sys.stdout) as pbar:
            for inputs, targets in pbar:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                # 【核心修复】：测试集也使用 FP32 绝对精度推理，拒绝 NaN
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)

                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()

                for label, prediction in zip(targets, predicted):
                    if label == prediction:
                        class_correct[classes[label]] += 1
                    class_total[classes[label]] += 1

    avg_loss = test_loss / test_total
    overall_acc = 100. * test_correct / test_total

    print("==============================", flush=True)
    print("测试完成！总体结果:", flush=True)
    print(f"平均损失 (Loss): {avg_loss:.4f}", flush=True)
    print(f"总体准确率 (Accuracy): {overall_acc:.2f}%", flush=True)
    print("==============================", flush=True)
    print("各类别详细表现:", flush=True)

    for classname in classes:
        acc = 100. * class_correct[classname] / class_total[classname]
        correct_count = class_correct[classname]
        total_count = class_total[classname]
        print(f" - {classname:<12}: {acc:.2f}% ({correct_count}/{total_count})", flush=True)


if __name__ == '__main__':
    predict()