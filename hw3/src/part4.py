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
        # plt.imshow(img3),plt.show()
        
        # TODO: 2. apply RANSAC to choose best H
        all_H=[]
        threshold =5
        iter = 5000
        all_matches = len(matches)
        
        chooseMatch=4
        min_dist=99999
        kparray1=np.array([[int(kp1[matches[i].queryIdx].pt[0]),int((kp1[matches[i].queryIdx].pt[1]))] for i in range(all_matches)])
        kparray2=np.array([[int(kp2[matches[i].trainIdx].pt[0]),int(kp2[matches[i].trainIdx].pt[1])] for i in range(all_matches)])
        ones=np.ones((kparray1.shape[0],1))
        
        kparray2=np.concatenate((kparray2,ones),axis=1).transpose(1,0)
        kparray1=kparray1.transpose(1,0)
        minDistance=999999999
        maxinlier=0
        bestH=[]
        
        for iter in tqdm(range(iter)):
            
            pairs = [[(int(kp1[matches[i].queryIdx].pt[0]),int((kp1[matches[i].queryIdx].pt[1]))),(int(kp2[matches[i].trainIdx].pt[0]),int(kp2[matches[i].trainIdx].pt[1]))] for i in random.sample(range(len(matches)), 4)]
            pair1=np.array([pairs[i][0] for i in range(4)])
            pair2=np.array([pairs[i][1] for i in range(4)])
            
            flag=False
            for i in range(4):
                for j in range(4):
                    if (pair1[i,0]==pair1[j,0]  and i!=j) or (pair2[i,0]==pair2[j,0] and i!=j):
                        flag=True
            if flag==True:
                
                continue
            # print(pair2,pair1)
            H=solve_homography(pair2,pair1)
            

            
            kparray2_prime=  np.matmul(H,kparray2)
            
            nor=np.repeat(kparray2_prime[2,:].reshape(1,kparray2_prime.shape[1]),3,axis=0)
            kparray2_prime= np.divide(kparray2_prime,nor)
            
            kparray2_prime=kparray2_prime[0:2,:]
            dist=np.sqrt(np.average(np.square( np.abs(kparray1-kparray2_prime)),axis=0))
            temp_dist=dist.copy()
            dist[dist>threshold]=0
            dist[dist!=0]=1
            nb_inliner=np.sum(dist)

            
            if nb_inliner>maxinlier:
                maxinlier=nb_inliner
                best_H=H
                bestdist=temp_dist
                
        
        # TODO: 3. chain the homographies
        all_H.append(best_H)
        H_total=np.identity(3)
        for h in all_H:
            H_total=np.matmul(H_total,h)
        # TODO: 4. apply warping
        dst=warping(im2,dst,H_total,0,h_max,0,w_max,'b')
        
    out = dst
    
    return out

if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)