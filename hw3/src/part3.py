import numpy as np
import cv2
from utils import solve_homography, warping


if __name__ == '__main__':

    # ================== Part 3 ========================
    secret1 = cv2.imread('../resource/BL_secret1.png')
    secret2 = cv2.imread('../resource/BL_secret2.png')
    corners1 = np.array([[429, 337], [517, 314], [570, 361], [488, 380]])
    corners2 = np.array([[346, 196], [437, 161], [483, 198], [397, 229]])
    h, w, c = (500, 500, 3)
    dst = np.zeros((h, w, c))
    dst2 = np.zeros((h, w, c))
    x = np.array([[0, 0],
                  [w, 0],
                  [w, h],
                  [0, h]
                  ])
    

    # TODO: call solve_homography() & warping
    output3_1 = None
    output3_2 = None
    H=solve_homography(corners1,x)
    xmin=np.min(corners1[:,0]).astype(np.int16)
    xmax=np.max(corners1[:,0]).astype(np.int16)
    ymin=np.min(corners1[:,1]).astype(np.int16)
    ymax=np.max(corners1[:,1]).astype(np.int16)
            
    output3_1=warping(secret1,dst,H,0, h, 0, w,'b')
    
    H=solve_homography(corners2,x)
    xmin=np.min(corners2[:,0]).astype(np.int16)
    xmax=np.max(corners2[:,0]).astype(np.int16)
    ymin=np.min(corners2[:,1]).astype(np.int16)
    ymax=np.max(corners2[:,1]).astype(np.int16)
    output3_2=warping(secret2,dst2,H,0, h, 0, w,'b')
    output3_3=np.sum(np.abs(output3_1-output3_2))
    

    cv2.imwrite('output3_1.png', output3_1)
    cv2.imwrite('output3_2.png', output3_2)