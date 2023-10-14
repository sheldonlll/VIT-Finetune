from transformers import Trainer
from transformers import ViTForImageClassification
import torch
import os
import random
import numpy as np
from utils.dataloader import DataGenerator, detection_collate
from torch.utils.data import DataLoader
from transformers import TrainingArguments
import torch
from datasets import load_metric
from transformers import ViTImageProcessor, get_cosine_schedule_with_warmup, AdamW, get_linear_schedule_with_warmup
from torchvision import transforms
from lion_pytorch import Lion

# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ])

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
metric = load_metric("accuracy")


def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)


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


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


classes_path = './model_data/cls_classes.txt'
class_names = get_classes(classes_path)
num_classes = len(class_names)
print(f"num_classes: {num_classes}")
seed = 10101 # 42
seed_everything(seed)

with open(r"./cls_train.txt", "r") as f:
    lines = f.readlines()

input_shape = [224, 224, 3]
np.random.shuffle(lines)
num_val = int(len(lines) * 0.1)
num_train = len(lines) - num_val
Batch_size = 16  # 网络训练每次要喂入多少的数据

train_dataset = DataGenerator(input_shape, lines[:num_train])
val_dataset = DataGenerator(input_shape, lines[num_train:], False)
gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                    drop_last=True, collate_fn=detection_collate)
gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                        drop_last=True, collate_fn=detection_collate)

labels = class_names

model_name_or_path = "./model_data/vit-base-patch16-224-in21k/"
output_dir = "./model_data/vit-base-patch16-224-in21k-finetune"
processor = ViTImageProcessor.from_pretrained(model_name_or_path)
model = ViTForImageClassification.from_pretrained(
    model_name_or_path,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)}
)
torch.nn.init.xavier_uniform_(model.classifier.weight)
torch.nn.init.constant_(model.classifier.bias, 0)

training_args = TrainingArguments(
  output_dir=output_dir,
  per_device_train_batch_size=16,
  evaluation_strategy="steps",
  num_train_epochs=16,
  fp16=True,
  save_steps=100,
  eval_steps=100,
  logging_steps=10,
  learning_rate=2e-4,
  save_total_limit=2,
  remove_unused_columns=False,
  push_to_hub=False,
  report_to='tensorboard',
  load_best_model_at_end=True,
)

# 优化器
# optimizer = AdamW(model.parameters(), lr=2e-4)
# optimizer = Lion(model.parameters(), lr=2e-4) # 《Symbolic Discovery of Optimization Algorithms》

# 学习率调度器
# scheduler = get_cosine_schedule_with_warmup(#很难收敛
#     optimizer, 
#     num_warmup_steps=100, 
#     num_training_steps=43056
# )

# scheduler = get_linear_schedule_with_warmup(
#     optimizer=optimizer,
#     num_warmup_steps=500,
#     num_training_steps=43056,
#     last_epoch=-1
# )

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=processor,
    # optimizers=(optimizer, scheduler)
)

trainer.train()

# ---------------


# ------------ 仿照原来的代码。逐段微调的方式
# Data Augmentation
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(10),
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ])

if False:
    training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=16,
    evaluation_strategy="steps",
    num_train_epochs=5,
    fp16=True,
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    learning_rate=2e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to='tensorboard',
    load_best_model_at_end=True,
    )


    # 训练数据的总批次数num_training_steps = num_train_epochs * num_batches_per_epoch
        # num_train = len(lines) - num_val; 
        # num_batches_per_epoch = num_train // Batch_size;

    num_training_steps = 26910 / 2
    num_warmup_steps = 100  # 学习率预热的步数

    # 优化器
    # optimizer = AdamW(model.parameters(), lr=2e-4)
    optimizer = Lion(model.parameters(), lr=2e-4, weight_decay=1e-2) # 《Symbolic Discovery of Optimization Algorithms》

    # 学习率调度器
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_training_steps
    )
    # 先冻结原始vit结构中的encoder和layernorm，只使用它来微调最后的classifier以及最前面的embeddings部分
    for param in model.vit.encoder.parameters():
        param.requires_grad = False
    for param in model.vit.layernorm.parameters():
        param.requires_grad = False

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor,
        optimizers=(optimizer, scheduler)
    )

    trainer.train()

if False:
    # 解冻所有权重
    for param in model.parameters():
        param.requires_grad = True

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=16,
        evaluation_strategy="steps",
        num_train_epochs=10,
        fp16=True,
        save_steps=100,
        eval_steps=100,
        logging_steps=10,
        learning_rate=2e-4,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to='tensorboard',
        load_best_model_at_end=True,
    )


    # 训练数据的总批次数num_training_steps = num_train_epochs * num_batches_per_epoch
        # num_train = len(lines) - num_val; 
        # num_batches_per_epoch = num_train // Batch_size;

    num_training_steps = 26910
    num_warmup_steps = 100  # 学习率预热的步数

    # 优化器
    # optimizer = AdamW(model.parameters(), lr=2e-4)
    optimizer = Lion(model.parameters(), lr=2e-4, weight_decay=1e-2) # 《Symbolic Discovery of Optimization Algorithms》

    # 学习率调度器
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_training_steps
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor,
        optimizers=(optimizer, scheduler)
    )

    trainer.train()



