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
#



AD_pretrained_model_dir='.exps/models/epoch_15000_Sub.pt'


#from network.c3d import C3D



def get_args():
	parser = argparse.ArgumentParser(description="PyTorch Video Classification Parser")
	# io /home/dgx-shared/anomaly_dataset/Anomaly-Videos
    # /home/dgx-shared/anomaly_dataset/Train
	
	

	parser.add_argument('--save_dir', type=str, default="features_MF", help="set logging file.")

	# optimization
	parser.add_argument('--batch_size', type=int, default=128,
						help="batch size")

	# model
	parser.add_argument('--model_type',
						type=str,
						default='mfnet',
						help="type of feature extractor",
						choices=['c3d', 'i3d', 'mfnet'])
    
	parser.add_argument('--pretrained_3d',
                        default='pretrained/MFNet3D_UCF-101_Split-1_96.3.pth',
                        #pretrained/MFNet3D_UCF-101_Split-1_96.3.pth
                        #c3d.pickle
						type=str,
						help="load default 3D pretrained model.")
    
    #parser.add_argument('--model_path', type=str, default="./exps/models/epoch_15000_Sub.pt", help="model path")
    
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

    #clip = sorted(glob(join('data', clip_name, '*.png')))
    #test = io.imread(clip[0])
    #rtest = resize(test, output_shape=(112, 200), preserve_range=True)
    clip = np.array([resize(frame, output_shape=(224, 224), preserve_range=True) for frame in clip_list])
    #clip = clip[:, :, 88:88 + 244, :]  # crop centrally

    # if verbose:
    #     clip_img = np.reshape(clip.transpose(1, 0, 2, 3), (112, 16 * 112, 3))
    #     io.imshow(clip_img.astype(np.uint8))
    #     io.show()

    clip = clip.transpose(3, 0, 1, 2)  # ch, fr, h, w
    clip = np.expand_dims(clip, axis=0)  # batch axis
    clip = np.float32(clip)
    print('Clip shape .........................................................')
    print(clip.shape)
    return torch.from_numpy(clip)

def single_extraction(input_clips_f, network: bool = None) -> int:
    
    input_clips = get_clip(input_clips_f, verbose=True)
    input_clips_t=np.array(input_clips_f)
    input_clips=torch.from_numpy(np.array(input_clips_f))
    transforms=build_transforms()
    input_clips =transforms(input_clips)

    random_seed = 1
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    device = torch.device("cuda" if torch.cuda.is_available()
                              else "cpu")
    
    print("Now doing the 3D network")
        #c3d_network = C3D(pretrained=pretrained_3d)
        #c3d_network.to(device)

        #from C3D_model import C3D
        #from network.model import static_model
        #from network.anomaly_detector_model import AnomalyDetector, RegularizedLoss, custom_objective

    model_type = args.model_type    
    pretrained_3d = args.pretrained_3d
    
    network = load_feature_extractor(model_type, pretrained_3d, device).eval()

    #X = Variable(input_clips)
    #X = X.cuda()
    X=input_clips
    X=X.unsqueeze(0)
    print('Size of X tensor')
    print(X.shape)
    #X=X.cuda()
    #input_clip=torch.from_numpy(input_clip)

    with torch.no_grad():
        outputs = network(X.to(device)).detach().cpu().numpy()

        features_writer = FeaturesWriter()
        start_frame=0
        vid_name='test_single_run_output'
        #c3d_outputs[0] as this is for a single use meaning no need to loop over the results
        features_writer.write(feature=outputs[0], video_name=vid_name, start_frame=start_frame, dir="test")
        
        print(len(outputs[0]))
        
        avg_segments=outputs[0]#dump()

    if network == None:
        return outputs,avg_segments, device, network
    else:
        return outputs,avg_segments


def single_prediction(model_path,features,lengths=16):
    

    if device == None:
        device = torch.device("cuda" if torch.cuda.is_available()
                              else "cpu")

    network = TorchModel(AnomalyDetector()) 
    
    model = network.load_model(model_path)
    
    model.eval()
    cudnn.benchmark = True
    model.to(device)
            
    features=torch.from_numpy(features)
    features = features.to(device)
    
    y_trues = torch.tensor([])
    y_preds = torch.tensor([])

    with torch.no_grad():
        
        features = features.to(device)
        outputs = model.forward(features).squeeze(-1)  # (batch_size, 32)
        print(outputs.shape)
            
    

    #print(y_true)
    print(outputs)
    #print("it is over")
    return outputs


def testing(clip_size=16,video_input='a7.mp4'):
    
    
    cap = cv2.VideoCapture(video_input)
    fourcc = "mp4v"

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter("test.mp4", cv2.VideoWriter_fourcc(*fourcc), fps, (W, H))

    frames = []
    while True:
        ret, frame = cap.read()

        frame = cv2.resize(frame, (224, 224))


        if not ret:
            print('Breaking video record ...')
            break

        frames.append(frame)

        if len(frames) == clip_size:
            
            outputs,avg_segments=single_extraction(frames)
            print("3D method completed")
            print(outputs.shape, avg_segments.shape )
            for frame in frames:
                out.write(frame)
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    #AD
    ad_model_dir = args.model_path
    
    

    # if sigle_y_pred.any() !=batch_y_pred.any():
    #     print("Error preditions from batch and sigle run are not the same?")
    #     print("b_y_pred =" + str(batch_y_pred))
    #     print("s_y_pred =" + str(sigle_y_pred))



if __name__ == '__main__':
    args = get_args()    

    testing()