import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config import *
from dataset import get_train_val_dataloaders
from model import get_model


def train():
    train_loader, val_loader, _ = get_train_val_dataloaders()

    model = get_model()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR_MAX, steps_per_epoch=len(train_loader), epochs=EPOCHS
    )

    scaler = torch.amp.GradScaler('cuda')
    best_val_acc = 0.0

    print("开始训练 (完美重现基线版本: ResNet34 + FP32 验证修复 NaN)...", flush=True)
    print("-" * 50, flush=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        current_lr = optimizer.param_groups[0]['lr']

        with tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
                  file=sys.stdout) as pbar:
            for inputs, targets in pbar:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                optimizer.zero_grad()

                # 仅在训练阶段使用混合精度加速
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                scheduler.step()

                train_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()

                step_loss = loss.item()
                step_acc = 100. * train_correct / train_total
                pbar.set_postfix_str(f"loss={step_loss:.4f}, acc={step_acc:.2f}%")

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = 100. * train_correct / train_total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                # 【核心修复】：移除这里的 autocast！全面采用 FP32 进行精确推理，避免 nan
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = 100. * val_correct / val_total

        print(f"Summary Epoch {epoch} (LR: {current_lr:.6f}):", flush=True)
        print(f"  [Train] Loss: {epoch_train_loss:.4f} | Acc: {epoch_train_acc:.2f}%", flush=True)
        print(f"  [Val  ] Std_Loss: {epoch_val_loss:.4f} | Acc: {epoch_val_acc:.2f}%", flush=True)

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                       "best_model.pth")
            print(f"  >>> 🚀 验证准确率提升至 {best_val_acc:.2f}%, 模型已保存！", flush=True)

        print("-" * 50, flush=True)


if __name__ == '__main__':
    train()