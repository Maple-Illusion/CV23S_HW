
import numpy as np
import cv2
# from utils import solve_homography, warping

a = np.array([[749, 521], [883, 525], [883, 750], [750, 750]])
print(a[2,0])
# N = 4

# A = np.zeros((1,9))
# u = np.zeros((N,2)).astype(int)
# v = np.zeros((N,2)).astype(int)
# U = np.ones(N*2).astype(int)
# V = np.ones(N*2).astype(int)
# for q in range(N):
#     if q % 2 == 0:
#         U[2*q] = u[q,0]
#         U[2*q+1] = u[q,1]
#         V[2*q] = v[q,0]
#         V[2*q+1] = v[q,1]

# print(V)
# print(U)


# for i in range(N):
#     A_regist = np.zeros((1,9))
#     A_regist2 = np.zeros((1,9))

#     A_regist[0,0] = u[i,0]
#     A_regist[0,1] = u[i,1]
#     A_regist[0,2] = 1
#     A_regist[0,6] = -u[i,0]*v[i,0]
#     A_regist[0,7] = -u[i,1]*v[i,1]

#     A_regist2[0,3] = u[i,0]
#     A_regist2[0,4] = u[i,1]
#     A_regist2[0,5] = 1
#     A_regist2[0,6] = -u[i,0]*v[i,0]
#     A_regist2[0,7] = -u[i,1]*v[i,1]
#         # print(A_regist)
#     B = np.concatenate((A_regist,A_regist2),axis=0)
#     # print(B)
#     A = np.concatenate((A,B),axis=0)
# A = np.delete(A,0,axis=0)
# print(A)