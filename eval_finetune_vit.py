import torch
from transformers import ViTForImageClassification
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from utils.dataloader import DataGenerator, detection_collate
import random
import os
import numpy as np
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def collate_fn(batch):
    return {
        'pixel_values': torch.stack([transform(torch.from_numpy(img).float()) for img, y in batch]),
        'labels': torch.tensor([y for img, y in batch])
    }


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names



if __name__ == '__main__':
    seed_everything(42)
    classes_path = './model_data/cls_classes.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    labels = class_names
    # 1. 加载模型
    model_path = "./model_data/vit-base-patch16-224-in21k-finetune/checkpoint-13900"
    model = ViTForImageClassification.from_pretrained(model_path,
        num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)}  
    )
    model.eval()  # 设置为评估模式

    # 2. 准备数据
    Batch_size = 16  # 网络训练每次要喂入多少的数据
    input_shape = [224, 224, 3]
    with open("./cls_test.txt","r", encoding="utf-8") as f:
        lines = f.readlines()
    val_dataset = DataGenerator(input_shape, lines, False)
    gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                            drop_last=True, collate_fn=collate_fn)
    val_loader = gen_val

    # 3. 进行评估
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            # inputs, labels = batch
            inputs = batch['pixel_values']
            labels = batch['labels']
            outputs = model(inputs)
            _, preds = torch.max(outputs.logits, 1)
            
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())

            accuracy = accuracy_score(true_labels, pred_labels)
            print(f"Accuracy: {accuracy * 100:.2f}% {i} / {len(val_loader)}")

    # 4. 计算准确率
    accuracy = accuracy_score(true_labels, pred_labels)
    print(f"Accuracy: {accuracy * 100:.2f}%")
