import os

# 获取当前文件夹的路径
current_directory = "./datasets/test/"

file_paths = []
for dirpath, dirnames, filenames in os.walk(current_directory):
    for file in filenames:
        file_paths.append(os.path.join(dirpath, file))

# 打印结果
lines = []
with open("./cls_test.txt", "r", encoding='utf-8') as f:
    lines = f.readlines()
lines = [lines[i].split(";")[0] for i in range(len(lines))]

text = []
for line, file_path in zip(lines, file_paths):
    text.append(line + ";" + file_path + "\n")

text = "".join(text)
with open("./clc_test_copy_path.txt", "w", encoding="utf-8") as f:
    f.write(text)