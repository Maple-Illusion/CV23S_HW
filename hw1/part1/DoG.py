import numpy as np
import cv2

class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_keypoints(self, image):
        ### TODO ####
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)
        point = []
        key = 0
        for u in range(self.num_octaves):
            gaussian_images = []
            h,w = image.shape
            image = cv2.resize(image,(int(w/(2*u if u > 0 else 1)),int(h/(2*u if u > 0 else 1))),interpolation= cv2.INTER_NEAREST)
            # h_p,w_p = image.shape
            blur = image
            # print(blur.shape)
            

            for i in range(4):
                
                blur = cv2.GaussianBlur(blur,(0,0),(self.sigma)**(i+1))
                
                
                # print("Blur size:", blur.shape)
                gaussian_images.append(blur)
                # cv2.namedWindow('blur',cv2.WINDOW_AUTOSIZE)
                # cv2.imshow('blur',np.uint8(gaussian_images[i]))
                
                # cv2.waitKey()
                # cv2.destroyWindow('blur')
            
            # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
            # - Function: cv2.subtract(second_image, first_image)
            dog_images = []
            dog_images.append(np.subtract(image,gaussian_images[0]))
            # cv2.imwrite(r'C:\Users\ataraxia\Documents\AV_LAB\NTU_CV\hw1_material\part1\DoG_pic'+'\\octave_'+str(u+1)+'_DoG_'+str(1)+'.png',np.uint8(dog_images[0]))
            for i in range(3):
                dog_images.append(np.subtract(gaussian_images[i],gaussian_images[i+1]))
                # cv2.imwrite(r'C:\Users\ataraxia\Documents\AV_LAB\NTU_CV\hw1_material\part1\DoG_pic'+'\\octave_'+str(u+1)+'_DoG_'+str(i+2)+'.png',np.uint8(dog_images[i+1]))
            # print(len(dog_images))
            ###### Normalize
            scale = 255
            dog_norm_image =[]
            for t in range(4):
                dog_norm = (dog_images[t] - np.min(dog_images[t]))/(np.max(dog_images[t])-np.min(dog_images[t]))
                dog_norm_image.append(dog_norm*scale)
                # np.savetxt(str(t+1)+'_P.txt',np.uint8(dog_norm*scale))
                # cv2.imwrite(r'C:\Users\ataraxia\Documents\AV_LAB\NTU_CV\hw1_material\part1\DoG_pic'+'\\octave_'+str(u+1)+'_DoG_'+str(t+1)+'.png',np.uint8(dog_norm_image[t]))


        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #         Keep local extremum as a keypoint
            layers = dog_norm_image
            structure = []
            structure = np.dstack(layers)
            # structure = np.concatenate(layers,dtype=np.float32,axis=1)
            
            h_p,w_p,lay = structure.shape
            # print(h_p,w_p,lay)
            ## kernel size 3*3*3
            ### w h num_pic steps
            
            # print(layers[0])
            
            # print(list(range(1,w_p-1)))
            for b in range(1,lay-1):
                for c in range(1,h_p-1): #### y
                    for j in range(1,w_p-1): ### x
                        
                        layer = structure[:,:,b] ## Full area
                        # print(layer.shape)
                        target = layer[c][j]
                        # print(target)
                        # print(layer)
                        # print(layer.shape)
                        Group = []
                        Group.append(structure[c-1:c+2,j-1:j+2,b-1])
                        # print(Group)
                        Group.append(structure[c-1:c+2,j-1:j+2,b])
                        # print(Group)
                        Group.append(structure[c-1:c+2,j-1:j+2,b+1])
                        # print(Group)
                        np.dstack(Group)
                        
                        # print(np.array(Group))
                        # print(np.max(Group))
                        if target == np.max(Group):
                            Group[1][1][1] = 0
                            if (target - np.max(Group)) >= self.threshold:
                                key += 1
                                # print('Got it: ',j,c)
                                point.append([j*(u+1),c*(u+1)])
                                
                                # keypoints = np.dstack(point)
                        elif target == np.min(Group):
                            Group[1][1][1] = 255
                            if -(target - np.min(Group)) >= self.threshold:
                                key += 1
                                # print('Got it: ',j,c)
                                point.append([j*(u+1),c*(u+1)])
                             
                                # keypoints = np.dstack(point)
                        elif len(np.unique(Group)) == 1:
                            # print('All the same')
                            key += 1
                            # print('Got it: ',j,c)
                            point.append([j*(u+1),c*(u+1)])
                            

                    # print(layer.shape)
                        # break
                    # break
                # break
        # keypoints = keypoints.swapaxes(1,0)
        # Step 4: Delete duplicate keypoints
        # - Function: np.unique
        
        point = np.array(point)
        # print(point.shape)
        keypoints = np.unique(point,axis=0)
        
        # print(keypoints.shape)
        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))]
        # print(key)
        # print(keypoints)
        return keypoints
