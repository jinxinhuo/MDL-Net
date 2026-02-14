import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# 引入我们写好的 MDL_Model
from models import MDL_Model

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from tqdm import tqdm

import os
import cv2 as cv
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torch.utils.data.sampler import Sampler
import cv2
import torch
from dataset.transform import xception_default_data_transforms

def compute_and_show_difference_batch(images, outputs):
    """
    计算批量图片与模型输出的像素差异，并返回差异图张量。
    :param images: 输入的原始图像张量 (batch_size, 3, 256, 256)
    :param outputs: 模型输出的图像张量 (batch_size, 3, 256, 256)
    :return: 差异图张量 (batch_size, 3, 256, 256)
    """
    images = images.cpu().detach().numpy().transpose(0, 2, 3, 1)  # 转换为 (batch_size, 256, 256, 3)
    outputs = outputs.cpu().detach().numpy().transpose(0, 2, 3, 1)  # 同上

    batch_size = images.shape[0]
    diff_images = []

    for i in range(batch_size):
        img1_array = images[i]
        img2_array = outputs[i]

        # 计算像素差异（取绝对值）
        diff_array = np.abs(img1_array - img2_array)

        # 将像素值归一化到 [0, 1]
        diff_array = np.clip(diff_array, 0, 1)

        diff_images.append(diff_array)

    # 转换为张量格式并调整形状
    diff_images = np.stack(diff_images, axis=0)  # (batch_size, 256, 256, 3)
    diff_images = torch.tensor(diff_images).permute(0, 3, 1, 2)  # 转换为 (batch_size, 3, 256, 256)

    return diff_images

CONFIG = {
    "seed": 42,
    "data_path": "./data/total",  # 已替换为通用路径
    "epochs": 4,
    "batch_size": 8,
    "learning_rate": 0.0001,
    "num_workers": 2,
    "model_name": "best_model.pth",
    "resume_training": False,
    "saved_model_path": "./output/best_model.pth",  # 已替换为通用路径
    "output_path": "./output/"  # 已替换为通用路径
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, train_loader, optimizer, epoch):
    model.train()
    metrics = {'loss_F1': 0.0, 'loss_ce': 0.0, 'loss_segce': 0.0, 'train_loss': 0.0}
    corrects, seg_corrects = 0, 0
    alpha, beta = 1.0, 0.1

    criterion_cls = nn.CrossEntropyLoss()
    criterion_seg = nn.CrossEntropyLoss()

    pbar = tqdm(train_loader, desc=f'Training Epoch {epoch + 1}', unit='batch')
    for img, label, mask, bi_mask in pbar:
        image = img.float().to(device)
        label = label.long().to(device)  # 分类标签转 Long
        mask = mask.float().to(device)
        bi_mask = bi_mask.long().to(device)

        optimizer.zero_grad()

        # 模型推理，传入 mask 触发 Guided Attention
        logits, re_img, segmentation_output = model(image, mask=mask)

        # 准确率统计
        predicted = torch.argmax(logits, dim=1)
        corrects += torch.sum(predicted == label).item()
        seg_pred = torch.argmax(segmentation_output, dim=1)
        seg_corrects += (seg_pred == bi_mask).sum().item()

        # 损失计算
        diff_images = compute_and_show_difference_batch(img, re_img).to(device)
        loss_F1 = F.l1_loss(mask, diff_images)
        loss_ce = criterion_cls(logits, label)
        loss_segce = criterion_seg(segmentation_output, bi_mask)

        loss = alpha * loss_ce + beta * loss_F1 + loss_segce

        loss.backward()
        optimizer.step()

        # 记录累加
        metrics['loss_F1'] += loss_F1.item()
        metrics['loss_ce'] += loss_ce.item()
        metrics['loss_segce'] += loss_segce.item()
        metrics['train_loss'] += loss.item()

    num_batches = len(train_loader)
    num_samples = len(train_loader.dataset)

    avg_metrics = {k: [v / num_batches] for k, v in metrics.items()}
    detection_acc = corrects / num_samples
    seg_acc = seg_corrects / (num_samples * 256 * 256)

    print(f'Train Loss: {avg_metrics["train_loss"][0]:.4f} | Det Acc: {detection_acc:.4f} | Seg Acc: {seg_acc:.4f}')
    return avg_metrics


@torch.no_grad()
def validate(model, val_loader):
    model.eval()
    val_loss, val_corrects, seg_corrects = 0.0, 0, 0
    total_pixels = 0
    alpha, beta = 1.0, 0.1

    criterion_cls = nn.CrossEntropyLoss()
    criterion_seg = nn.CrossEntropyLoss()

    for img, label, mask, bi_mask in tqdm(val_loader, desc='Validating', unit='batch'):
        image = img.float().to(device)
        label = label.long().to(device)
        mask = mask.float().to(device)
        bi_mask = bi_mask.long().to(device)

        logits, re_img, segmentation_output = model(image, mask=mask)

        # 准确率统计
        predicted = torch.argmax(logits, dim=1)
        val_corrects += torch.sum(predicted == label).item()

        seg_pred = torch.argmax(segmentation_output, dim=1)
        seg_corrects += (seg_pred == bi_mask).sum().item()
        total_pixels += bi_mask.numel()

        # 损失计算
        diff_images = compute_and_show_difference_batch(image, re_img).to(device)
        loss = alpha * criterion_cls(logits, label) + \
               beta * F.l1_loss(mask, diff_images) + \
               criterion_seg(segmentation_output, bi_mask)

        val_loss += loss.item()

    epoch_loss = val_loss / len(val_loader)
    detection_acc = val_corrects / len(val_loader.dataset)
    seg_acc = seg_corrects / total_pixels

    print(f'Val Loss: {epoch_loss:.4f} | Det Acc: {detection_acc:.4f} | Seg Acc: {seg_acc:.4f}')
    return epoch_loss


def plot_loss_history(loss_history, num_epochs):
    plt.figure(figsize=(10, 6))
    epochs_range = range(1, num_epochs + 1)

    plt.plot(epochs_range, loss_history['loss_F1'], label='Loss F1', color='blue')
    plt.plot(epochs_range, loss_history['loss_ce'], label='Loss CE', color='green')
    plt.plot(epochs_range, loss_history['loss_segce'], label='Loss Seg CE', color='red')
    plt.plot(epochs_range, loss_history['train_loss'], label='Total Loss', linestyle='--', color='black')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(CONFIG['output_path'], 'loss_curve.png'))
    plt.show()


def main():
    torch.manual_seed(CONFIG['seed'])
    torch.cuda.manual_seed_all(CONFIG['seed'])
    os.makedirs(CONFIG['output_path'], exist_ok=True)

    train_dataset = read_mask_01_label_data(root_dir=os.path.join(CONFIG['data_path'], "train"), data_type="train")
    val_dataset = read_mask_01_label_data(root_dir=os.path.join(CONFIG['data_path'], "val"), data_type="val")

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True,
                              num_workers=CONFIG['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False,
                            num_workers=CONFIG['num_workers'])

    model = MDL_Model(num_classes=2, pretrained=True).to(device)

    if CONFIG['resume_training'] and os.path.exists(CONFIG['saved_model_path']):
        model.load_state_dict(torch.load(CONFIG['saved_model_path']))
        print(f"Resumed training from {CONFIG['saved_model_path']}")

    optimizer = optim.SGD(model.parameters(), lr=CONFIG['learning_rate'], momentum=0.9, weight_decay=5e-4)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'], eta_min=1e-6)

    best_val_loss = float('inf')
    loss_history = {'loss_F1': [], 'loss_ce': [], 'loss_segce': [], 'train_loss': []}

    for epoch in range(CONFIG['epochs']):
        print(f'\nEpoch {epoch + 1}/{CONFIG["epochs"]}\n' + '-' * 20)

        # Train
        epoch_metrics = train(model, train_loader, optimizer, epoch)
        for k in loss_history.keys():
            loss_history[k].extend(epoch_metrics[k])

        # Val
        val_loss = validate(model, val_loader)
        scheduler.step()

        # Save Best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(CONFIG['output_path'], CONFIG['model_name'])
            torch.save(model.state_dict(), save_path)
            print(f"-> Best model saved! (Val Loss: {best_val_loss:.4f})")

    plot_loss_history(loss_history, CONFIG['epochs'])


if __name__ == "__main__":
    main()