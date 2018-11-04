# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 23:34:29 2018

@author: Borak_Madi
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 12:59:37 2018

@author: Borak_Madi

This build we will based on the paper A Fast and Robust Homography Scheme for Real-time Planar Target Detection 
Hamid Bazaragani- Olexa Bilaniuk- Robert Langaiere ..


Detector : FAST9
Descriptor : BRIEF
Homography Estimation : RHO algorthim .

We will made feew changes , they select the area or deisred area of extract features manually in our project we will select this area depending of the result of MOSSE filter .
 
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 21:34:25 2018

@author: Borak_Madi

"""



import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import copy
from utils import get_four_points

global flag_show, detector_Binary,tracker_type,detector_type,Match_methods
global flag_compare 
## These Control Variables 

## FLAGS _ 
flag_compare = False ## flag to show comparable images
flag_show_matches = False  ## flag to show mtches between reference image and matches
flag_get_result = False  ## get the result as video 
flag_show = True  ## show the result tracking of MOSSE + Homography Tracking  !! without compare 
flag_draw_keypoint = True  # draw keypoints for the Mask ! :D            

## Paramters of algorithim of Estimate Homography
algo_Method = cv2.RHO
threshold = 8# default is 3.0 is used for RANSAC and RHO only
MaxIter = 2020 # default number of iterations to run
confidence_t = 0.995 # the confidence level 
GOOD_MATCH_PERCENT =1 ## our solutions build from 


   
 ## Extract and Detect features
detector_type = ''
descriptor_type = ''
detector = None
descriptor = None
detector_Binary = True


tracker_type = 'MOSSE'

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
flag_pick_bb = True;
flag_pick_pionts = True


####

   
 
###    
fps = 0
fps_avg = 0 # frames each iterations 
count = 0 # number of iterations 
scale_percent = 50


Match_methods =" " 
global mask
global resolution_frames ,pts_new


# convert mask with 4 points as polygon 
def define_poly_mask(points):
      

     p1 =points[0][0]
     p2 = points[0][1]
     p3 = points[0][2]
     p4 = points[0][3]
     mask = np.zeros(resolution_frames, np.uint8)
     points = np.array([p1, p2, p3, p4])
     points = np.int32(points)
#     cv2.convexHull(points,points)
     cv2.fillConvexPoly(mask, points, 255)
     if(flag_show==True):
        cv2.imshow('Maks of Polygon',mask)
    
     return mask
   

# convert the bbox to  rectangle mask 
def define_mask_with_bb(bbox):
    
    mask = np.zeros((resolution_frames), dtype=np.uint8)
    mask[int(bbox[1]):int(bbox[1])+int(bbox[3]),int(bbox[0]):int(bbox[0])+int(bbox[2])] = 255
    if(flag_show==True):
        cv2.imshow('Maks of Rectangle ',mask)
#        cv2.waitKey(0)
#        cv2.destroyWindow('Maks of ROI')
    return  mask

# rectified or perspective correction to our image ;
# this function called once for the first frame
def point_ply (img):
    
    if (flag_pick_pionts == True):
            pts_src = get_four_points(img)
            flag_pick_pionts == False

    return pts_src
    
#  detect keypoints and extract feature on image inside the mask!
def image_detect_and_compute(detector_t,descriptor_t,img,mask=None,):
    """Detect and compute interest points and their descriptors."""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp  =   detector_t.detect(img,mask)
    kp, des= descriptor_t.compute(img, kp) 
    return img, kp, des
    

#  return model-or hypothesis with RHO or RANSAC algorithim    
def  create_model(des_prev,kp_prev,des_now,kp_now):
    
     matches = bf.match(des_prev,des_now)
         # Sort matches by score
     matches.sort(key=lambda x: x.distance, reverse=False)  
     # for more speed-up we can take portion of all matches 
     numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
     matches = matches[:numGoodMatches]
    # Extract location of good matches
     points1 = np.zeros((len(matches), 2), dtype=np.float32)
     points2 = np.zeros((len(matches), 2), dtype=np.float32)
     
     for i, match in enumerate(matches):
         points1[i, :] = kp_prev[match.queryIdx].pt
         points2[i, :] = kp_now[match.trainIdx].pt
     
     M, mask = cv2.findHomography(points1, points2, algo_Method,ransacReprojThreshold=threshold,maxIters=MaxIter,confidence=confidence_t)  
     return M,mask
    
# draw the 4-vertcies polygon that surround that the desired object
# pts_prev is old 4 points from previous image ! 
def draw_image_polygon(img_new,Homography_model,pts_prev):
    
    img_poly = img_new.copy()   
    pts_new = cv2.perspectiveTransform(pts_prev,Homography_model)    
    cv2.polylines(img_poly,np.int32([pts_new]),True,255,3, cv2.LINE_AA)
    return img_poly,pts_new
   
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split(".")
 
if __name__ == '__main__' :
 
    
##  We can add  more  values to these arrays :D
 # we produce 3-tuple (Descriptor_type,Detector_type,Threshould) => example ('SIFT','FAST',3.0) this is 3-tuple !!

## Example of  Arrays of Detectors/Descriptors/Threshould     
#    threshold_picks = [0.5, 1, 2,3,5,7,8,9,10,11,15,17,19,20]
#     detectors_types = ['FAST''ORB', 'AGAST','AKAZE', 'BRISK', 'SURF', 'SIFT', 'GFTTDetector'] # almost all of them 
#     descriptors_types = ['BoostDesc','BRIEF','DAISY','FREAK','SIFT','SURF'] # we can add more !

    detectors_types = ['FAST']
    descriptors_types = ['BRIEF','ORB']
    threshold_picks = [6,3]
  
    
    ## create track_model Use MOSSE !
#    tracker = cv2.TrackerMOSSE_create() # we can define once !
# 
#    
    for detector_type in detectors_types: 
        
        
        tracker = cv2.TrackerMOSSE_create()
        
        if detector_type == 'FAST' :
            detector = cv2.FastFeatureDetector_create()
        if detector_type == 'ORB':
            detector = cv2.ORB_create() 
        if detector_type =='AGAST':
            detector = cv2.AgastFeatureDetector_create()    
        if detector_type == 'SURF':
            detector = cv2.xfeatures2d.SURF_create()
        if detector_type == 'SIFT':
            detector = cv2.xfeatures2d.SIFT_create()
         
               
            
        for descriptor_type in descriptors_types:
            detector_Binary = True
            if descriptor_type == 'BRIEF' :
                descriptor = cv2.xfeatures2d.BriefDescriptorExtractor_create()
                Match_methods = "B.F Hamming Norm"
                
            if descriptor_type == 'BRISK' :
                descriptor = cv2.BRISK_create()
                Match_methods = "B.F Hamming Norm"   
            
            if descriptor_type == 'ORB' :
               descriptor = cv2.ORB_create() 
               Match_methods = "B.F Hamming Norm"  
            if descriptor_type == 'FREAK' :
               descriptor = cv2.xfeatures2d.FREAK_create()
               Match_methods = "B.F Hamming Norm"  
               
            
            for threshold in threshold_picks: 
                
                  
                tracker = cv2.TrackerMOSSE_create()
                print('Thresh = ',threshold)
                print('Detector = ',detector_type)
                print('Desc = ',descriptor_type)
                  # Read video
                video = cv2.VideoCapture('2Ad.MOV')
                ######
                # Exit if video not opened.
                if not video.isOpened():
                    print("Could not open video")
                    sys.exit()
                
                
                
                ######################################################
                #                  Read first frame.
                #######################################################
                # for first frame we need to create bounding box for MOSSE filter , and click 
                 # on the 4 points in the frame to create polygon 
                     
            
                ok, frame = video.read()
                if not ok:
                    print('Cannot read video file')
                    sys.exit()   
            
            
                    
                ###############################
                #resize the frame 
                ##############################
                #percent of original size
                width = int(frame.shape[1] * scale_percent / 100)
                height = int(frame.shape[0] * scale_percent / 100)
                dim = (width, height)
            
                frame=  cv2.resize(frame,dim,cv2.INTER_CUBIC)
                resolution_frames = frame.shape[0],frame.shape[1]  
            
         
                ###########################
                # Get 4-points polygon by clicks of mouse 
                ###########################
                
                pts_poly = point_ply(frame)
                
                print(type( pts_poly))
                print(pts_poly.shape)
                print(pts_poly)
             
    
                pts_poly = np.array([pts_poly])
                
                print(type( pts_poly))
                print( pts_poly.shape)
                print(pts_poly)
#               

          
                
                
                if(flag_show == True):
                    img_ply,pts_new = draw_image_polygon(frame,np.identity(3),pts_poly)
                    print('the bbox points ' , pts_poly)
                    print('Detector Type ', type(detector))
                    print('Descriptor Type ', type(descriptor))
                    print('new points ' , pts_new)
                    cv2.imshow('first Image with polygon',img_ply)
                    cv2.waitKey(0)
                    cv2.destroyWindow('first Image with polygon')
                    
#                    
#                    
                    
                ###############################    
                #  MOSSE - Start BBOX
                ##############################
#                bbox = (186, 74, 223, 328) # it's set manually 
                if(flag_pick_bb==True):
#                 maually with picking point done by use with mouse-handler
                    bbox = cv2.selectROI(frame, False)
#                    flag_pick_bb = False
                                    
                if(flag_show == True):
                    
                     # Draw bounding box
                    img_bbox = frame.copy()
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    cv2.rectangle(img_bbox, p1, p2, (255,0,0), 2, 1)
                    cv2.imshow('first Image with bbox',img_bbox )
                    cv2.waitKey(0)
                    cv2.destroyWindow('first Image with bbox')     
##        
##        
                ###############################
                #  Fetch keypoints and build descriptos over the first frame !
                #  * We can add a parameters for each method - detectors or descriptors
                ###############################
                
                if detector_type == 'FAST' :
                     detector.setThreshold(10)
                    # Print all default params
                     print("Threshold: ", detector.getThreshold())
                     print("nonmaxSuppression: ", detector.getNonmaxSuppression())
                     print("neighborhood: ", detector.getType())
                    
                if (detector_type is 'SIFT') and (descriptor_type is 'ORB') :
                    continue
                else:
                    
                    
                    
                    frame_gray, kp_prev, des_prev =  image_detect_and_compute(detector,descriptor,frame,define_mask_with_bb(bbox))   
#                    frame_gray, kp_prev, des_prev =  image_detect_and_compute(detector,descriptor,frame)   
                    ## draw keypoint on the mask region or the desired region we get from bbox of MOSSE !
                    if(flag_draw_keypoint == True):   
                        img_with_keypoints = cv2.drawKeypoints(frame, kp_prev, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                        cv2.imshow('Draw Keypoints',img_with_keypoints)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
    
                # Initialize tracker with first frame and bounding box
                
                ok = tracker.init(frame, bbox)
                cv2.destroyAllWindows()
                    
#                   
                while True:
#
                    # read frame 
                    ok, frame = video.read()
#                
                    if not ok: # if it's finished reading :D 
                        break
#                    
                    # Start timer
                    timer = cv2.getTickCount()
#                 
                    # resize the frame !
                    frame = cv2.resize(frame,dim,cv2.INTER_CUBIC)
                    
                    
                    ##########################
                    # MOSSE UPDATE 
                    #########################
                    # Update tracker , this new bbox will be the region where we extract 
                    # our Detector and Descriptors 
                    ok, bbox= tracker.update(frame)
                    
                    
#                    ##########################
#                     Next-Frame  : Extract & Detect features
#                    #########################
                    mask_region = define_mask_with_bb(bbox)
                    frame_gray, kp_now, des_now =  image_detect_and_compute(detector,descriptor,frame,mask_region)
#                    frame_gray, kp_now, des_now =  image_detect_and_compute(detector,descriptor,frame)
                    if(flag_draw_keypoint == True):   
                        img_with_keypoints = cv2.drawKeypoints(frame, kp_now, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                        cv2.imshow('Draw Keypoints',img_with_keypoints)
#                        print("Number of key points ", len(kp_prev))
                      
                        
                        
                        
                    Homography_model,mask = create_model(des_prev,kp_prev,des_now,kp_now)
                    
                    if(Homography_model is not None):
                       
                        img_ply,pts_new = draw_image_polygon(frame,Homography_model,pts_poly)               
                        if(flag_show == True):
                               p1 = (int(bbox[0]), int(bbox[1]))
                               p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                               cv2.rectangle(img_ply, p1, p2, (255,0,0), 2, 1)
                        cv2.putText(img_ply, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
                        cv2.imshow('Tracking with Homography',img_ply)                          
                    #################
                    # We need to save the previous descriptor and detectors when we have solution or model  
                        kp_prev = kp_now
                        des_prev = des_now
                        pts_poly =pts_new
                    else:
                        cv2.putText(img_ply, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)  
                    
                  
                              
                   # Calculate Frames per second (FPS)
                    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
                    fps_avg  = fps_avg + fps
                    count = count + 1
                    k = cv2.waitKey(1) & 0xff
                    if k == 27 : 
                        cv2.destroyAllWindows()
                        break 
                
                
                
              ############ print all the   
                
                
                # end of while loop
                if count != 0:             
                    print(["The Average FPS : ",fps_avg/count ])
                else :
                    print(["None - FPs : ",0 ])   
                     
                fps_avg = 0;
                fps = 0;
                count= 0;
                # When everything done, release the video capture and video write objects
                # END OF THRESHHOLD loop
                
            # END OF DESCriptors loop
    
        # END OF Detectors loop
              
              
        video.release()
        cv2.destroyAllWindows()
    
#
