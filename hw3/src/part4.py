import numpy as np
import cv2
from matplotlib import pyplot as plt
import random
from tqdm import tqdm
from utils import solve_homography, warping

random.seed(669258)

def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3)
    out = None

    # for all images to be stitched:
    for idx in tqdm(range(len(imgs)-1)):
        im1 = imgs[idx]
        im2 = imgs[idx + 1]

        # TODO: 1.feature detection & matching
        orb = cv2.ORB_create()
        kp1 = orb.detect(im1,None)
        kp2 = orb.detect(im2,None)
        kp1, des1 = orb.compute(im1, kp1)
        kp2, des2 = orb.compute(im2, kp1)
        # img2 = cv2.drawKeypoints(im1, kp1, None, color=(0,255,0), flags=0)
        # plt.imshow(img2), plt.show()
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1,des2)
        matches = sorted(matches, key = lambda x:x.distance)
        img3 = cv2.drawMatches(im1,kp1,im2,kp2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
 
        plt.imshow(img3),plt.show()
        # TODO: 2. apply RANSAC to choose best H

        # TODO: 3. chain the homographies

        # TODO: 4. apply warping

    return out

if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)