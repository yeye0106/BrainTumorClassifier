import os
import torch

DATA_DIR = "./brain-tumor-mri-dataset"
TRAIN_DIR = os.path.join(DATA_DIR, "Training")
TEST_DIR = os.path.join(DATA_DIR, "Testing")

IMG_SIZE = 224
BATCH_SIZE = 128
EPOCHS = 30
LR_MAX = 1e-3         # 恢复 OneCycleLR 配合的高效学习率
NUM_WORKERS = 8

CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
NUM_CLASSES = len(CLASSES)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")