import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from transformers import ViTForImageClassification

class Top1_Classification:
    def __init__(self, model_path, classes_path, input_shape, cuda=True):
        self.model = ViTForImageClassification.from_pretrained(model_path)
        self.model.eval()
        if cuda:
            self.model.cuda()
        self.cuda = cuda

        with open(classes_path) as f:
            self.class_names = [c.strip() for c in f.readlines()]

        self.input_shape = input_shape
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((input_shape[0], input_shape[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def detect_image(self, image):
        image = np.array(image)
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
            image = np.repeat(image, 3, axis=-1)
        image = image[:, :, :3]
        photo = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            if self.cuda:
                photo = photo.cuda()
            outputs = self.model(photo)
            preds = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
        
        arg_pred = np.argmax(preds)
        class_name = self.class_names[arg_pred]
        probability = np.max(preds)
        
        return arg_pred, class_name, probability

# Initialize your model
model_path = "./model_data/vit-base-patch16-224-in21k-finetune/checkpoint-13900"
classes_path = "./model_data/cls_classes.txt"
input_shape = [224, 224, 3]

top1_classifier = Top1_Classification(model_path, classes_path, input_shape)

# Evaluate Top-1 Accuracy
correct = 0
total = 0
with open("./cls_train.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
import random
random.shuffle(lines)
lines = lines[:2000]
total = len(lines)
for index, line in enumerate(lines):
    annotation_path = line.split(';')[1].strip()
    x = Image.open(annotation_path)
    y = int(line.split(';')[0])

    pred, _, _ = top1_classifier.detect_image(x)
    correct += (pred == y)

    if index % 100 == 0:
        print(f"[{index} / {total}] Current accuracy: {correct / total * 100:.2f}%")

final_accuracy = correct / total * 100
print(f"Top-1 Accuracy: {final_accuracy:.2f}%")
