import numpy as np
import cv2
import argparse
import os
import pandas as pd
from JBF import Joint_bilateral_filter


def main():
    parser = argparse.ArgumentParser(description='main function of joint bilateral filter')
    parser.add_argument('--image_path', default='./testdata/1.png', help='path to input image')
    parser.add_argument('--setting_path', default='./testdata/1_setting.txt', help='path to setting file')
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ### TODO ###
    # params = pd.read_csv(args.setting_path,sep=',',header=None,encoding='utf-8')
    # params = np.genfromtxt(args.setting_path,delimiter=',',names=True,encoding='utf-8')
    with open(args.setting_path,encoding='utf-8') as F:
        for _ in F.readlines():
            line  = (_.replace('\n','')).split(',')
            # print(line)
    

if __name__ == '__main__':
    main()