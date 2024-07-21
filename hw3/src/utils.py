import numpy as np
from scipy.interpolate import NearestNDInterpolator

def solve_homography(u, v):
    """
    This function should return a 3-by-3 homography matrix,
    u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
    :param u: N-by-2 source pixel location matrices
    :param v: N-by-2 destination pixel location matrices
    :return:
    """
    N = u.shape[0]
    H = None

    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')

    # TODO: 1.forming A
    A = np.empty((1,8))
    # print(A)
    U = np.ones(N*2).astype(int)
    V = np.ones(N*2).astype(int)
    for q in range(N):
        U[2*q] = u[q,0]
        U[2*q+1] = u[q,1]
        V[2*q] = v[q,0]
        V[2*q+1] = v[q,1]

    for i in range(N):
        A_regist = np.zeros((1,8))
        A_regist2 = np.zeros((1,8))

        A_regist[0,0] = u[i,0]
        A_regist[0,1] = u[i,1]
        A_regist[0,2] = 1
        A_regist[0,6] = -u[i,0]*v[i,0]
        A_regist[0,7] = -u[i,1]*v[i,0]
        # A_regist[0,8] = 1
        A_regist2[0,3] = u[i,0]
        A_regist2[0,4] = u[i,1]
        A_regist2[0,5] = 1
        A_regist2[0,6] = -u[i,0]*v[i,1]
        A_regist2[0,7] = -u[i,1]*v[i,1]
        # A_regist2[0,8] = 1
        # print(A_regist)
        B = np.concatenate((A_regist,A_regist2),axis=0)
        A = np.concatenate((A,B),axis=0)
    A = np.delete(A,0,axis=0)
    # print(A)

    # print(V)
    # TODO: 2.solve H with A
    H = np.linalg.solve(A,V)
    H = np.concatenate((H,[1]),axis=0).reshape(3,3)
    # print(H.shape)
    return H


def warping(src, dst, H, ymin, ymax, xmin, xmax, direction='b'):
    """
    Perform forward/backward warpping without for loops. i.e.
    for all pixels in src(xmin~xmax, ymin~ymax),  warp to destination
          (xmin=0,ymin=0)  source                       destination
                         |--------|              |------------------------|
                         |        |              |                        |
                         |        |     warp     |                        |
    forward warp         |        |  --------->  |                        |
                         |        |              |                        |
                         |--------|              |------------------------|
                                 (xmax=w,ymax=h)

    for all pixels in dst(xmin~xmax, ymin~ymax),  sample from source
                            source                       destination
                         |--------|              |------------------------|
                         |        |              | (xmin,ymin)            |
                         |        |     warp     |           |--|         |
    backward warp        |        |  <---------  |           |__|         |
                         |        |              |             (xmax,ymax)|
                         |--------|              |------------------------|

    :param src: source image
    :param dst: destination output image
    :param H:
    :param ymin: lower vertical bound of the destination(source, if forward warp) pixel coordinate
    :param ymax: upper vertical bound of the destination(source, if forward warp) pixel coordinate
    :param xmin: lower horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param xmax: upper horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param direction: indicates backward warping or forward warping
    :return: destination output image
    """

    h_src, w_src, ch = src.shape
    h_dst, w_dst, ch = dst.shape
    H_inv = np.linalg.inv(H)

    # TODO: 1.meshgrid the (x,y) coordinate pairs
    x_dst = np.array(range(xmin,xmax))
    y_dst = np.array(range(ymin,ymax))
    dx,dy = np.meshgrid(x_dst,y_dst)
    # print(len(dx),len(dy))
    x_src = np.array(range(w_src))
    y_src = np.array(range(h_src))
    sx,sy = np.meshgrid(x_src,y_src)
    
    # TODO: 2.reshape the destination pixels as N x 3 homogeneous coordinate
    one_d = np.ones(dx.shape)
    dst_m = np.vstack((dx,dy,one_d)).reshape(3,-1).astype(np.int16)
    one_s = np.ones(sx.shape)
    src_m = np.vstack((sx,sy,one_s)).reshape(3,-1).astype(np.int16)
    # print(src_m)

    if direction == 'b':
        H_inv = np.linalg.inv(H)
        # TODO: 3.apply H_inv to the destination pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        V = H_inv @ dst_m
        # xy = V[0:2][:]
        
        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of source image)
        # xy_new = np.round(xy[0:2][:]/V[2][:]).astype(np.int16)
        dil=np.repeat(V[2,:].reshape(1,V.shape[1]),3,axis=0)
        xy_new= np.around(np.divide(V,dil)[0:2,:]).astype(np.int16)
        # TODO: 5.sample the source image with the masked and reshaped transformed coordinates
        x_new = xy_new[0,:]
        y_new = xy_new[1,:]
        x_new[x_new>=w_src] = 0
        x_new[x_new<0]=0

        y_new[y_new>=h_src] = 0
        y_new[y_new<0]=0 ###region done

        dst_id=np.stack((dx,dy)).transpose(1,2,0).reshape(-1,2)
        dy_id =dst_id[:,1]
        dx_id =dst_id[:,0]
       

        dy_id[y_new==0]=0
        dx_id[x_new==0]=0
        # print(dy_id)
        # TODO: 6. assign to destination image with proper masking
        dmask = np.zeros(dst.shape)
        smask = np.ones(dst.shape)

        weight_dst=1-1/(w_src+h_src)*(x_new+y_new)
        weight_src=1/(w_src+h_src)*(x_new+y_new)

        dmask[(dy_id,dx_id)]=np.stack((weight_dst,weight_dst,weight_dst),axis=1)
        smask[(dy_id,dx_id)]=np.stack((weight_src,weight_src,weight_src),axis=1)
        dmask[dst==0]=0
        smask[dst==0]=1
        dst[(dy_id,dx_id)]=src[(y_new,x_new)]
        pass

    elif direction == 'f':
        # TODO: 3.apply H to the source pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        U = H @ src_m
        # print(U)
        # print(U.shape)
        xy = U[0:2][:]
        # print(len(xy[1,:]))
        
        # print(xy_new)
        # z = np.hypot(xy[0,:],xy[1,:])
        # inp = NearestNDInterpolator(list(zip(xy[0,:],xy[1,:])),z)
        # print(inp)
        # print(list(zip(src_m[1][:],src_m[0][:])))
        # print(dst_id)
        

        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of destination image)
        xy_new = np.round(xy[0:2][:]/U[2][:]).astype(np.int16)
        # TODO: 5.filter the valid coordinates using previous obtained mask

        # TODO: 6. assign to destination image using advanced array indicing
        dst[(xy_new[1,:],xy_new[0,:])]=src[(sy.flatten(),sx.flatten())]
        pass

    return dst
