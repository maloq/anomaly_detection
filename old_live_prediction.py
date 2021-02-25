import sys
# insert at 1, 0 is the script path (or '' in REPL)
import os
home=os.getcwd()
from data_loader import VideoIter
import torch
import argparse
import cv2

from utils.utils import build_transforms

import pickle

from feature_extractor import FeaturesWriter
from tqdm import tqdm

from network.anomaly_detector_model import AnomalyDetector, RegularizedLoss, custom_objective

from features_loader import FeaturesLoaderVal

import torch.backends.cudnn as cudnn
from network.TorchUtils import *

from network.anomaly_detector_model import AnomalyDetector

from features_loader import FeaturesLoaderEval
import numpy as np

from torch.autograd import Variable
from utils.load_model import load_feature_extractor
from skimage.transform import resize

#pretrained_3d="pretrained/MFNet3D_UCF-101_Split-1_96.3.pth"

model_path='exps/models/epoch_15000_Sub.pt'

def get_args():
    
    parser = argparse.ArgumentParser(description="PyTorch Video Classification Parser")
    parser.add_argument('--save_dir', type=str, default="features_MF", help="set logging file.")

    parser.add_argument('--model_type', type=str, default='mfnet', help="type of feature extractor", choices=['c3d', 'i3d', 'mfnet'])
    
    parser.add_argument('--pretrained_3d', default='pretrained/MFNet3D_UCF-101_Split-1_96.3.pth', type=str, help="load default 3D pretrained model.")
    
    return parser.parse_args()

def get_clip(clip_list, verbose=True):
    """
    Loads a clip to be fed to NN for classification.

    Parameters
    ----------
    clip_name: str
        the name of the clip (subfolder in 'data').
    verbose: bool
        if True, shows the unrolled clip (default is True).

    Returns
    -------
    Tensor
        a pytorch batch (n, ch, fr, h, w).
    """

    clip_list = np.array([frame for frame in clip_list])
    clip = np.array([resize(frame, output_shape=(224, 224), preserve_range=True) for frame in clip_list])
    
    #clip = clip.transpose(3, 0, 1, 2)  # ch, fr, h, w
    clip = np.expand_dims(clip, axis=0)  # batch axis
    clip = np.uint8(clip)
    #print('Clip shape .........................................................')
    #print(clip.shape)
    return torch.from_numpy(clip)

def single_extraction(input_clips_f, network=None, device=None):
    
    input_clips = get_clip(input_clips_f, verbose=True)
    
    #input_clips_t=np.array(input_clips_f)
    #input_clips=torch.from_numpy(np.array(input_clips_f))
    
    transforms=build_transforms(mode='mfnet')
    if DEBUG:
        print('Clip shape')
        print(input_clips[0].shape)
    input_clips =transforms(input_clips[0])

    random_seed = 1
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    

    
    X=input_clips
    X=X.unsqueeze(0)
    print('Size of X tensor')
    print(X.shape)
    

    with torch.no_grad():
        outputs = network(X.to(device)).detach().cpu().numpy()

        features_writer = FeaturesWriter(num_videos = 1)
        start_frame=0
        vid_name='test_single_run_output'
        features_writer.write(feature=outputs[0], video_name=vid_name, idx=0, dir="test")
        
        print(len(outputs[0]))
        
        cnn_feature=outputs[0]#dump()

    if network == None:
        return outputs,cnn_feature, device, network
    else:
        return outputs,cnn_feature


def single_prediction(features,lengths=16, device = None, model = None, save_path= 'test', time = 0):
    
    
    features = features.to(device)
    
    y_trues = torch.tensor([])
    y_preds = torch.tensor([])

    with torch.no_grad():
        
        features = features.to(device)
        outputs = model.forward(features).cpu().numpy()
        
        print('Prediction is over')
        print(type(outputs))
        print(outputs)
    
    if DEBUG:
        #print('video length: ', length)        
        print('preds length: ', len(outputs))      
        
    with open('{}.txt'.format(os.path.join('test', 'test')), "a") as o:            
        for n in outputs:
            o.write("".join(str(n)) + " " + str(time) + "\n")
        
    return outputs


def testing(clip_size=16,video_input='a8.mp4', mean_num = 4, ad_model_input = 6144):    

    cap = cv2.VideoCapture(video_input)
    fourcc = "mp4v"
    fps = int(cap.get(cv2.CAP_PROP_FPS))    
    out = cv2.VideoWriter("test.mp4", cv2.VideoWriter_fourcc(*fourcc), fps, (224, 224))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    
    #init 3D model
    model_type = args.model_type    
    pretrained_3d = args.pretrained_3d    
    network = load_feature_extractor(model_type, pretrained_3d, device).eval()
    #init AD model
    ad_network = TorchModel(AnomalyDetector())    
    ad_model = ad_network.load_model(model_path)
    ad_model.eval()
    cudnn.benchmark = True
    ad_model.to(device)
    
    with open('{}.txt'.format(os.path.join('test', 'test')), "w") as o:            
        o.write("")
            
    frames = []
    i =  0
    time = 0
    cnn_features = np.zeros((mean_num, ad_model_input))
    
    while True:
        ret, frame = cap.read()
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #print('video frame size: ', (W,H))
        
        try:
            frame = cv2.resize(frame, (224, 224))
        except Exception as e:
            print(str(e))
            pass
        
        if not ret:
            print('Breaking video record ...')
            break

        frames.append(frame)

        if len(frames) > clip_size-1:
              
            outputs,cnn_feature=single_extraction(frames, network, device)
            print("3D method completed")
            print(cnn_feature.shape)
            for frame in frames:
                out.write(frame)
            frames = []
            time+=16/fps
            
            if i == mean_num-1:
                cnn_features[i] = cnn_feature
                avg_features = torch.from_numpy(cnn_features.mean(axis=0)).float()
                print('cnn_features  ', cnn_features.shape)
                print('avg_features  ', avg_features.shape)

                outputs = single_prediction(avg_features, lengths=16, device=device, model=ad_model, time=time)
                
                i=0
            else:
                cnn_features[i] = cnn_feature
                i+=1
            
            
    cap.release()
    out.release()    

    # if sigle_y_pred.any() !=batch_y_pred.any():
    #     print("Error preditions from batch and sigle run are not the same?")
    #     print("b_y_pred =" + str(batch_y_pred))
    #     print("s_y_pred =" + str(sigle_y_pred))

if __name__ == '__main__':
    args = get_args()    
    DEBUG = False

    testing()