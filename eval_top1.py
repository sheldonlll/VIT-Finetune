import numpy as np
import torch
import os.path
from PIL import Image
from torch.autograd import Variable
from sklearn.metrics import precision_score, recall_score, f1_score
from classification import Classification, _preprocess_input, VITClassification
from utils.utils import letterbox_image


class top1_Classification(Classification):# top1_Classification(Classification)
    def detect_image(self, image):        
        crop_img = letterbox_image(image, [self.input_shape[0],self.input_shape[1]])
        photo = np.array(crop_img,dtype = np.float32)

        # 图片预处理，归一化
        photo = np.reshape(_preprocess_input(photo),[1,self.input_shape[0],self.input_shape[1],self.input_shape[2]])
        photo = np.transpose(photo,(0,3,1,2))

        with torch.no_grad():
            photo = Variable(torch.from_numpy(photo).type(torch.FloatTensor))
            if self.cuda:
                photo = photo.cuda()
            preds = torch.softmax(self.model(photo)[0], dim=-1).cpu().numpy()

        arg_pred = np.argmax(preds) # 最大可能性的类别所对应的索引
        probability = np.max(preds)  # 最大可能性的类别的可能性大小
        class_name = self.class_names[np.argmax(preds)] # 最大可能性的类别所对应的名称
        return arg_pred,probability,class_name
    
    def detect_image_vit(self, image):
        image = np.array(image)
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
            image = np.repeat(image, 3, axis=-1)
        image = image[:, :, :3]
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        photo = transform(image).unsqueeze(0)
        with torch.no_grad():
            if self.cuda:
                photo = photo.cuda()
            outputs = self.model(photo)
            preds = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
        arg_pred = np.argmax(preds) # 最大可能性的类别所对应的索引
        class_name = self.class_names[arg_pred]
        probability = np.max(preds)
        return arg_pred, class_name, probability  # 返回类名和概率


def preprocess_input(x,):
    x /= 127.5
    x -= 1.
    return x


# fw = open(r"C:\Users\GM\Desktop\cout\top1yes.csv", 'w')
import shutil

def copy_file(source_path, destination_path):
    shutil.copy(source_path, destination_path)

def evaluteTop1(classfication, lines: list):
    correct = 0
    total = len(lines)
    
    # fw = open(r"C:\Users\psy\Desktop\analysis/mobilenet.txt", 'w') #创建一个文档，用于存放预测错误的图片信息
    for index, line in enumerate(lines):
        annotation_path = line.split(';')[1].replace('\n', '').replace('渭', 'μ')
        # if os.path.exists(annotation_path) == False and os.path.exists(annotation_path.replace("test", "train")) == True:
        #     copy_file(annotation_path.replace("test", "train"), annotation_path)
        # if os.path.exists(annotation_path) == False:
        #     print(f"pop element: {annotation_path}")
        #     lines.pop(index)
        #     with open("./cls_test.txt", "w", encoding="utf-8") as f:
        #         f.write("".join(lines))
        #     continue
        x = Image.open(annotation_path)
        y = int(line.split(';')[0])


        pred = pro = name = None
        if isinstance(classfication, VITClassification) == False:
            pred = classfication.detect_image(x)[0]

            pro = classfication.detect_image(x)[1]
            name = classfication.detect_image(x)[2]
        else:
            pred = classfication.detect_image_vit(x)[0]
            pro = classfication.detect_image_vit(x)[1]
            name = classfication.detect_image_vit(x)[2]

        # if y == pred: #判断是否预测错误，预测错误就写入文档中
        #     print(os.path.basename(annotation_path))  # 输出预测错误的图像名称
        #     t = os.path.basename(annotation_path)
        #     fw.write(t)
        #     fw.write(",")
        #     fw.write(str(y))
        #     fw.write(",")
        #     fw.write(str(name))
        #     fw.write(",")
        #     fw.write(str(pro))
        #     fw.write("\n")

        correct += pred == y
        if index % 100 == 0:
            print(f"[{index} / {total}] {correct / total}")
    return correct / total

classfication = top1_Classification()
with open("./cls_test.txt","r", encoding="utf-8") as f:
    lines = f.readlines()
top1 = evaluteTop1(classfication, lines)
print("top-1 accuracy = %.2f%%" % (top1*100))

# def evaluteTop1(classfication, lines):
#     y_true = []
#     y_pred = []
#     total = len(lines)

#     for index, line in enumerate(lines):
#         annotation_path = line.split(';')[1].replace('\n', '')
#         x = Image.open(annotation_path)
#         y = int(line.split(';')[0])

#         pred, _, _ = classfication.detect_image(x)

#         y_true.append(y)
#         y_pred.append(pred)

#         if index % 100 == 0:
#             print("[%d/%d]" % (index, total))

#     return y_true, y_pred

