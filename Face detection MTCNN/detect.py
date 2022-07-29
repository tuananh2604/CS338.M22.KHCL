import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import time
from scipy import spatial
from numpy import dot
from numpy.linalg import norm


frame_size = (640,480)
IMG_PATH = 'face1'
DATA_PATH = 'face1'

def trans(img):
    transform = transforms.Compose([
            transforms.ToTensor()
        ])
    return transform(img)

def load_faceslist():
    if device == 'cpu':
        embeds = torch.load(DATA_PATH+'/faceslistCPU.pth')
    else:
        embeds = torch.load(DATA_PATH+'/faceslist.pth')
    names = np.load(DATA_PATH+'/usernames.npy')
    return embeds, names

# def euclide_dt(a,b):
#     norm_diff = a-b
#     norm_diff = torch.sum(torch.pow(norm_diff, 2), dim=1)
#     print(a.reshape(512))
#     print(b[:,:,2].reshape(512))
#     return norm_diff

def cosine_dt(a,b):
    _,_,n=b.shape
    norm_diffs=[]
    for i in range(n):
        # print(a.reshape(512))
        # print(b[:,:,i].reshape(512))
        # print(torch.dot(a.reshape(512) , b[:,:,i].reshape(512)))
        # print(a.norm(dim=1, p=0)[0,0])
        # print(b[:,:,i].reshape(1,512,1).norm(dim=1, p=0)[0,0])
        # print(a.norm(dim=1, p=0)[0,0] * b[:,:,i].reshape(1,512,1).norm(dim=1, p=0)[0,0])
        _a=a.reshape(512).cpu().detach().numpy()
        _b=b[:,:,i].reshape(512).cpu().detach().numpy()
        norm_diff = 1 - spatial.distance.cosine(_a, _b)
        # print(norm_diff)

        # norm_diffs.append((norm_diff*(10**7))**2)
        norm_diffs.append(norm_diff)
    norm_diffs=torch.tensor(norm_diffs).double()
    # print(norm_diffs)
    return norm_diffs.reshape(1,-1)

def inference(model, face, local_embeds, threshold = 3):
    #local: [n,512] voi n la so nguoi trong faceslist
    embeds = []
    # print(trans(face).unsqueeze(0).shape)
    embeds.append(model(trans(face).to(device).unsqueeze(0)))
    detect_embeds = torch.cat(embeds) #[1,512]
    # print(detect_embeds.shape)
                    #[1,512,1]                                      [1,512,n]
    # norm_diff = detect_embeds.unsqueeze(-1) - torch.transpose(local_embeds, 0, 1).unsqueeze(0)

    # norm_diff = 1 - spatial.distance.cosine(detect_embeds.unsqueeze(-1).cpu().detach().numpy(), torch.transpose(local_embeds, 0, 1).unsqueeze(0).cpu().detach().numpy())
    norm_diff=cosine_dt(detect_embeds.unsqueeze(-1),torch.transpose(local_embeds, 0, 1).unsqueeze(0))
    print(norm_diff,norm_diff.shape)
    # print(detect_embeds.unsqueeze(-1),detect_embeds.unsqueeze(-1).shape)
    # print(torch.transpose(local_embeds, 0, 1).unsqueeze(0),torch.transpose(local_embeds, 0, 1).unsqueeze(0).shape)
    # norm_score = torch.sum(torch.pow(norm_diff, 2), dim=1) #(1,n), moi cot la tong khoang cach euclide so vs embed moi?
    
    min_dist, embed_idx = torch.max(norm_diff, dim = 1)
    print(min_dist, names[embed_idx])
    print(min_dist.shape)
    print("``````````````````````````````````")

    if min_dist < 0.7:
        return -1, -1
    else:
        return embed_idx, min_dist.double()
def extract_face(box, img, margin=20):
    face_size = 160
    img_size = frame_size
    margin = [
        margin * (box[2] - box[0]) / (face_size - margin),
        margin * (box[3] - box[1]) / (face_size - margin),
    ] #tạo margin bao quanh box cũ
    box = [
        int(max(box[0] - margin[0] / 2, 0)),
        int(max(box[1] - margin[1] / 2, 0)),
        int(min(box[2] + margin[0] / 2, img_size[0])),
        int(min(box[3] + margin[1] / 2, img_size[1])),
    ]
    img = img[box[1]:box[3], box[0]:box[2]]
    face = cv2.resize(img,(face_size, face_size), interpolation=cv2.INTER_AREA)
    # face = Image.fromarray(face)
    return face

if __name__ == "__main__":
    prev_frame_time = 0
    new_frame_time = 0
    power = pow(10, 6)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = InceptionResnetV1(
        classify=False,
        pretrained="casia-webface"
    ).to(device)
    model.eval()

    mtcnn = MTCNN(thresholds= [0.7, 0.7, 0.8] ,keep_all=True, device = device)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    embeddings, names = load_faceslist()
    while cap.isOpened():
        isSuccess, frame = cap.read()
        if isSuccess:
            boxes, _ = mtcnn.detect(frame)
            if boxes is not None:
                for box in boxes:
                    bbox = list(map(int,box.tolist()))
                    face = extract_face(bbox, frame)
                    idx, score = inference(model, face, embeddings)
                    if idx != -1:
                        frame = cv2.rectangle(frame, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,0,255), 6)
                        score = torch.Tensor.cpu(score[0]).detach().numpy()
                        frame = cv2.putText(frame, names[idx] + ' {}%'.format(int(score*100)), (bbox[0],bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 2, 2)
                    else:
                        frame = cv2.rectangle(frame, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,0,255), 6)
                        # frame = cv2.putText(frame,'Unknown', (bbox[0],bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 2,2)

            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time
            fps = str(int(fps))
            cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_DUPLEX, 3, (100, 255, 0), 3, 2)

            cv2.imshow('Face Recognition', frame)
            if boxes is not None:
                cv2.imshow('face crop',face)
            if cv2.waitKey(1)&0xFF == 27:
                break
        # break

    cap.release()
    cv2.destroyAllWindows()
    #+": {:.2f}".format(score)