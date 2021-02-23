import argparse
import torch
import torch.backends.cudnn as cudnn
from network.TorchUtils import *

from network.anomaly_detector_model import AnomalyDetector

from features_loader import FeaturesLoaderEval
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from os import path
import numpy as np
import pytorch_wrapper as pw
import pickle
import os
import cv2
import matplotlib.pyplot as plt

DEBUG = True

def get_args():
    parser = argparse.ArgumentParser(
        description="PyTorch Video Classification Parser")
    
    parser.add_argument('--demo_videos_path', default="/home/dgx-shared/anomaly_dataset/Anomaly-Videos/",
                        help="folder with videos to be used for demo")
    #./fights_unsorted/
    #                    help="path to features")
    parser.add_argument('--features_path', default='features_MF',
                        help="path to features")
    parser.add_argument('--annotation_path', default="Demo_annotations.txt",
                        help="path to demo annotations")
    parser.add_argument('--model_path', type=str, default="./exps/models/epoch_15000_Sub.pt",
                        help="set logging file.")
    return parser.parse_args()


def get_predictions(model_path, annotation_path, features_path, device=None):

    data_loader = FeaturesLoaderEval(features_path=features_path,
                                     annotation_path=annotation_path)

    data_iter = torch.utils.data.DataLoader(data_loader,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=0,
                                            pin_memory=True)

    if device == None:
        device = torch.device("cuda" if torch.cuda.is_available()
                              else "cpu")

    network = TorchModel(AnomalyDetector()) 
    
    model = network.load_model(model_path)
    
    model.eval()
    cudnn.benchmark = True
    model.to(device)


    y_trues = torch.tensor([])
    y_preds = torch.tensor([])

    with torch.no_grad():
        for features, lengths in tqdm(data_iter):
            # features is a batch where each item is a tensor of 32 4096D features
            features = features.to(device)
            outputs = model.forward(features).squeeze(-1)  # (batch_size, 32)
            for vid_len, output in zip(lengths, outputs.cpu().numpy()):
                y_true = np.zeros(vid_len)
                y_pred = np.zeros(vid_len)

                segments_len = vid_len // 32

                for i in range(32):
                    segment_start_frame = i * segments_len
                    segment_end_frame = (i + 1) * segments_len
                    y_pred[segment_start_frame: segment_end_frame] = output[i]

                if y_trues is None:
                    y_trues = y_true
                    y_preds = y_pred
                else:
                    y_trues = np.concatenate([y_trues, y_true])
                    y_preds = np.concatenate([y_preds, y_pred])

    print("Prediction is over")
    return y_preds


def figure2opencv(figure):
    figure.canvas.draw()
    img = np.fromstring(figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(figure.canvas.get_width_height()[::-1] + (3,))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def GUI(video_path, y_pred, s_path="output_video"):
    DISPLAY_IMAGE_SIZE = 920
    BORDER_SIZE = 20
    FIGHT_BORDER_COLOR = (0, 0, 255)
    NO_FIGHT_BORDER_COLOR = (0, 255, 0)

    plot_range = 100
    videoReader = cv2.VideoCapture(video_path)
    isCurrentFrameValid, currentImage = videoReader.read()

    
    fps = videoReader.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(s_path+'_v5.avi', fourcc, fps, (960, 480))

    fig = plt.figure()
    frame_count = 0
    length = int(videoReader.get(cv2.CAP_PROP_FRAME_COUNT))
    
    resultVideo = []
    
    while isCurrentFrameValid:
        frame_count = frame_count + 1

        targetSize = DISPLAY_IMAGE_SIZE - 2 * BORDER_SIZE
        currentImage = cv2.resize(currentImage, (targetSize, targetSize))

        resultImage = cv2.copyMakeBorder(currentImage,
                                         BORDER_SIZE,
                                         BORDER_SIZE,
                                         BORDER_SIZE,
                                         BORDER_SIZE,
                                         cv2.BORDER_CONSTANT,
                                         value=NO_FIGHT_BORDER_COLOR)

        resultImage = cv2.resize(resultImage, (480, 480))

        plt.plot(frame_count, y_pred[frame_count-1], color='green',
                 marker='o', linestyle='-', linewidth=2, markersize=2)

        plt.ylim(-0.05, 1.05)
        if frame_count < 100:
            plt.xlim(0, 100)
        else:
            plt.xlim(frame_count-100, frame_count+100)

        plt.xlim(0, length)
        plot_img = figure2opencv(fig)
        plot_img = cv2.resize(plot_img, (480, 480))  # (0, 0), None, .25, .25)

        resultImage = np.concatenate((resultImage, plot_img), axis=1)

        
        print("saving ", s_path.split('/')[-1])
        
        out.write(resultImage)        

        
        print(str(frame_count)+"/"+str(length))
        
        
        #np.append(resultVideo, resultImage)

        isCurrentFrameValid, currentImage = videoReader.read()
        
    videoReader.release()
    out.release()


def to_frames(video_path):
    vidcap = cv2.VideoCapture(video_path)
    video_name = video_path.split('/')[-1].split('.')[0]
    success,image = vidcap.read()
    if success:
        os.mkdir(video_name, exist_ok=True)
    length = int(videoReader.get(cv2.CAP_PROP_FRAME_COUNT))
    
    video_frames = np.zeros(1)

    count = 0 
    
    while success:        
        np.append
        cv2.imwrite(os.path.join(video_name, "frame%d.jpg" % count ), image)     # save frame as JPEG file      
        success,image = vidcap.read()
    count += 1
    
    
if __name__ == '__main__':

    args = get_args()
    home = os.getcwd()

    annotation_path = args.annotation_path
    video_path_list = []

    with open(annotation_path, 'r') as f:
            lines = f.read().splitlines(keepends=False)
            for line in lines:
                file_name = line.split()[0]
                file_path = os.path.join(args.demo_videos_path, file_name)
                video_path_list.append(file_path)

    y_preds = get_predictions(
        args.model_path, args.annotation_path, args.features_path)    
             
    total_frames = 0
    print('total length: ', len(y_preds))
                                
    for video_path in video_path_list:

        cap = cv2.VideoCapture(video_path)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        save_path = os.path.join(home, video_path.split('/')[-1].split('.')[0])
        video_preds =  y_preds[total_frames:(total_frames+length)]

        print(save_path)
        
        if DEBUG:
            print('video length: ', length)        
            print('preds length: ', len(video_preds))
        
        with open('{}.txt'.format( os.path.join('preds', video_path.split('/')[-1].split('.')[0]) ), "w") as o:
            
            for n in video_preds:
                o.write("".join(str(n)) + "\n")
                
                
        GUI(video_path, video_preds , s_path=save_path)
        total_frames += length
