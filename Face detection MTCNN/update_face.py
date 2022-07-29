import glob
import torch 
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
import os
from PIL import Image
import numpy as np
import cv2

IMG_PATH = 'face1'
DATA_PATH = 'face1'

device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
def trans(img):
    transform = transforms.Compose([
            transforms.ToTensor()
        ])
    return transform(img)
    
model = InceptionResnetV1(
    classify=False,
    pretrained="casia-webface"
).to(device)

model.eval()

embeddings = []
names = []

for usr in os.listdir(IMG_PATH):
    embeds = []
    for file in glob.glob(os.path.join(IMG_PATH, usr)+'/*.jpg'):
        # print(usr)
        try:
            img =   cv2.imread(file)
        except:
            continue
        with torch.no_grad():
            # print('smt')
            embeds.append(model(trans(img).to(device).unsqueeze(0))) #1 anh, kich thuoc [1,512]
    if len(embeds) == 0:
        continue
    embedding = torch.cat(embeds).mean(0, keepdim=True) #dua ra trung binh cua 50 anh, kich thuoc [1,512]
    embeddings.append(embedding) # 1 cai list n cai [1,512]
    # print(embedding)
    names.append(usr)
    
embeddings = torch.cat(embeddings) #[n,512]
names = np.array(names)

if device == 'cpu':
    torch.save(embeddings, DATA_PATH+"/faceslistCPU.pth")
else:
    torch.save(embeddings, DATA_PATH+"/faceslist.pth")
np.save(DATA_PATH+"/usernames", names)
print('Update Completed! There are {0} people in FaceLists'.format(names.shape[0]))
print(embeddings)