import os

datapath = []
with open("./cls_test.txt", "r", encoding="utf-8", errors="replace") as f:
    datapath = f.readlines()

# datapath = [\
#     datapath[i][:datapath[i].find("test") + 5] + datapath[i][datapath[i].find("test") + 4:].split("/")[1].split(" ")[0] + \
#     datapath[i][datapath[i].find(datapath[i][datapath[i].find("test") + 4:].split("/")[1]) + len(datapath[i][datapath[i].find("test") + 4:].split("/")[1]):] \
#     for i in range(len(datapath)) \
# ]
newdatapath = []
path = "./datasets/diatom species datasets (seven data augmentation)/diatom species datasets (seven data augmentation)/test/"
for i in range(len(datapath)):
    cur_path = datapath[i].split(";")[0] + ";" + path + datapath[i].split("/")[-1].replace("test\\", "")
    newdatapath.append(cur_path)
datapath = "".join(newdatapath)
print(newdatapath[0])
with open("./cls_test.txt", "w", encoding="utf-8", errors="replace") as f:
    f.write(datapath)