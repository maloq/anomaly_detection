import os
import cv2
import argparse
import glob


parser = argparse.ArgumentParser(description="vidoe annotatate maker")

parser.add_argument('--dataset_path', default='./fights_unsorted/',
                    help="dataset_path paths")


parser.add_argument('--file_name', default="unsorted_annotations",
                    help="the name of the end annotation file")


# this is for the adding of new file for training or etc

def annotatate_file(dataset_path, file_name="Demo_annotations"):
    # path lenght start end
    # Fighting/Fighting047_x264.mp4 4459 Fighting 200 1830
    # Testing_Normal_Videos_Anomaly/Normal_Videos_872_x264.mp4 530

    if os.path.exists(file_name+".txt") == True:
        os.remove(file_name+".txt")
    file = open(file_name+".txt", "a")
    s = os.path.join(dataset_path, '*/*.mp4')
    path_list = glob.glob(s)

    for path in path_list:
        print(path)

        videoReader = cv2.VideoCapture(path)
        length = int(videoReader.get(cv2.CAP_PROP_FRAME_COUNT))

        dir = path.split('/')

        str1 = dir[-2] + "/" + dir[-1]+" "+str(length) + '\n'

        file.write(str1)

    file.close()
    home = os.getcwd()
    return home+"/"+file_name+".txt"


if __name__ == '__main__':
    args = parser.parse_args()
    annotatate_file(args.dataset_path, file_name=args.file_name)