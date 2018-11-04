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
flag_get_result = True  ## get the result as video 
flag_show = True  ## show the result tracking of MOSSE + Homography Tracking  !! without compare 


## Paramters of algorithim of Estimate Homography
algo_Method = cv2.RANSAC
threshold = 3.0  # default is 3.0 is used for RANSAC and RHO only
MaxIter = 2000 # default number of iterations to run
confidence_t = 0.995 # the confidence level 

   
   
fps_avg = 0 # frames each iterations 
count = 0 # number of iterations 
fps = 0
detector_Binary = False
MIN_MATCH_COUNT = 10
tracker_type = 'MOSSE'
detector_type = 'SURF'
MAX_FEATURES = 500

GOOD_MATCH_PERCENT = 0.15
scale_percent = 40
threshold = 1
Match_methods =" " 
global mask
global resolution_frames 


def show_compare_images(img1,img2):
#   numpy_horizontal = np.hstack((img1, img1))
   numpy_horizontal_concat = np.concatenate((img1, img2), axis=1)
    # Display tracker type on frame7
   if(flag_compare == True):
       cv2.putText(numpy_horizontal_concat, "RANSAC - Matching Using Mask ", (int(numpy_horizontal_concat.shape[1]/4) - 10 ,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255),2)
       cv2.putText(numpy_horizontal_concat, "RANSAC - Matching Using whole Image", (int(3*numpy_horizontal_concat.shape[1]/4)- 13 ,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255),2)
       cv2.putText(numpy_horizontal_concat, "Detector = " + detector_type , (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)
       cv2.putText(numpy_horizontal_concat,"Inliers Percent = " + str(GOOD_MATCH_PERCENT), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)
       cv2.putText(numpy_horizontal_concat,"Threshold = " + str(threshold), (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)
       cv2.putText(numpy_horizontal_concat,"Matching = " + Match_methods , (10,110), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)
       cv2.imshow('Numpy Horizontal Concat', numpy_horizontal_concat)
          # Display tracker type on frame
   else:
       cv2.putText(numpy_horizontal_concat, tracker_type+" Homography Tracking ", (int(numpy_horizontal_concat.shape[1]/4) - 10 ,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255),2)
       cv2.putText(numpy_horizontal_concat,  tracker_type+" Bounding-Box Tracking ", (int(3*numpy_horizontal_concat.shape[1]/4)- 13 ,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255),2)
       cv2.imshow('Numpy Horizontal Concat', numpy_horizontal_concat)
       
   return numpy_horizontal_concat

def define_mask_with_bb(bbox):
    
    mask = np.zeros((resolution_frames), dtype=np.uint8)
    mask[int(bbox[1]):int(bbox[1])+int(bbox[3]),int(bbox[0]):int(bbox[0])+int(bbox[2])] = 255
#    if(flag_show == True):
#        cv2.imshow("Mask you",mask)
#        cv2.waitKey(0)
    return  mask
    
def perspective_correction (im_src):
    
     
    # Destination image
    size = (300,400,3)

    im_dst = np.zeros(size, np.uint8)

    
    pts_dst = np.array(
                       [
                        [0,0],
                        [size[0] - 1, 0],
                        [size[0] - 1, size[1] -1],
                        [0, size[1] - 1 ]
                        ], dtype=float
                       )
    
    

    # Show image and wait for 4 clicks.
#    cv2.imshow("Image", im_src)
#    pts_src = get_four_points(im_src);
    pts_src = np.array([[219. , 92.],[372. , 98.],[368. ,358.],[184. ,353.]], np.int32)
#    cv2.destroyWindow("Image")
    
    # Calculate the homography
    h, status = cv2.findHomography(pts_src, pts_dst)

    # Warp source image to destination
    im_dst = cv2.warpPerspective(im_src, h, size[0:2])

#    # Show output
#    cv2.imshow("Rectified Image", im_dst)
#    cv2.waitKey(500)
#    cv2.destroyWindow("Rectified Image")
#    
    return im_dst,h
    
    
    

def image_detect_and_compute(detector, img,mask=None,):
    """Detect and compute interest points and their descriptors."""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if detector_type == 'FAST':
        kp  =  detector.detect(rectified_frame,None)
        brief = brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        kp, des= brief.compute(img, kp)
    else :
        kp, des = detector.detectAndCompute(img,mask)
    return img, kp, des
    

def draw_image_matches(detector, img1, img2,mask1=None,mask2=None, nmatches=10):
   
    img2_org = img2.copy()
    img1, kp1, des1 = image_detect_and_compute(detector, img1,mask1)
    img2, kp2, des2 = image_detect_and_compute(detector, img2,mask2)
    
    
    if(detector_Binary==False):
        
        # FLANN Parameters
        FLANN_INDEX_KDTREE =1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees =5)
        search_params = dict(checks=50) 
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.match(np.asarray(des1,np.float32),np.asarray(des2,np.float32))
    else:
        """Draw ORB feature matches of the given two images."""
        # HAMMING Distance 
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1,des2)
        

    
    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)  
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]
   
    if(flag_show_matches == True):
         img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None)
         plt.figure(figsize=(18, 18))
         plt.title(type(detector))
         plt.imshow(img3); plt.show()
         
    
    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
     
    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt
       
    # Find homography using percent of matches 
    
#    M, mask = cv2.findHomography(points1, points2, algor_estime,threshold)
    
    M, mask = cv2.findHomography(points1, points2, algo_Method,ransacReprojThreshold=threshold,maxIters=MaxIter,confidence=confidence_t)
    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    img2_org = cv2.polylines(img2_org,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        
    if(flag_show == True):     
            # Display tracker type on frame
        cv2.putText(img2_org, "Detector = " + detector_type , (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)
        cv2.putText(img2_org,"Inliers Percent = " + str(GOOD_MATCH_PERCENT), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)
        cv2.putText(img2_org,"Threshold = " + str(threshold), (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
        cv2.putText(img2_org,"Matching = " + Match_methods , (100,110), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
        cv2.imshow("Result", img2_org)
        
    return img2_org
        

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split(".")
 
if __name__ == '__main__' :
 
    # Set up tracker.
    # Instead of MIL, you can also use
 
    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
 
    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
        if tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create()
 
    
    
    # Set up Detectors Using  Main 2D Features framework - OpenCv (xfeatures2d)

    detectors_types = ['ORB', 'SIFT','SURF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    
    if detector_type == 'ORB':
        detector = cv2.ORB_create()
        detector_Binary = True
        Match_methods = "Brute Force - Hamming Distance "
    if detector_type == 'SIFT':
        detector  = cv2.xfeatures2d.SIFT_create()
        detector_Binary = False
        Match_methods = "FLANN : KD-Tree "
    if detector_type == 'SURF':
        minHessian = 400;
        detector = cv2.xfeatures2d.SURF_create(hessianThreshold=minHessian)
        Match_methods = "FLANN : KD-Tree "
        detector_Binary = False
    if detector_type == 'DAISY':
        detector  = cv2.xfeatures2d.DAISY_create()
    if detector_type == 'FAST':
        detector = cv2.FastFeatureDetector_create()
        detector_Binary = True
        


        
        
        ########### we need to add more detectors in future to compare between them !#########
        # https://docs.opencv.org/3.4.1/d7/d7a/group__xfeatures2d__experiment.html 
        #########################################################
        
         
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
         
    fps_avg = 0
    count = 0;
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
    frame = cv2.resize(frame,dim,cv2.INTER_CUBIC)
    resolution_frames = frame.shape[0],frame.shape[1]  
        
   
    
    ## EXAMPLE OF MASK parameter
#    mask1 = 255* np.ones(frame1.shape,dtype=np.uint8)
#    mask1[:,:int(mask1.shape[1]/2)] = 0
#    kp, des = detector.detectAndCompute(frame,mask = mask1)
#    img2 = cv2.drawKeypoints(frame,kp,None,(255,0,0),4)
#    cv2.imshow("Window",img2)
#    cv2.waitKey(0)
 
     
    
    ### UPDATED CODE - WE NEED to resize the video     
    if(flag_get_result==True) : 
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]
        # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
        if(flag_compare==True or flag_get_result==True) :
            out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (2*frame_width,frame_height))
        else:
            out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
    
    
    ###############################
    #  4-Ploygon && Perspective Correction 
    ##############################
    
    # Define an initial 4-points polygon and implement perespective correction
    rectified_frame,to_rect_h = perspective_correction(frame)
    
    
    
    ##***** we do this in second experment 
    ###############################
    #  Extract feature of rectifed image
    ##############################
    img = cv2.cvtColor(rectified_frame, cv2.COLOR_BGR2GRAY)
    if detector_type == 'FAST':
        kp  =  detector.detect(rectified_frame,None)
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        kp, des = brief.compute(img, kp)
    else:
        kp, des = detector.detectAndCompute(rectified_frame, None)
    
    img_kp = cv2.drawKeypoints(img, kp, img)
    if(flag_show_matches == True):
        plt.figure(figsize=(15, 15))
        plt.imshow(img_kp); plt.show()
    

    # Define an initial bounding box
    bbox = (287, 23, 86, 320)
    # Uncomment the line below to select a different bounding box
    bbox = (174, 70, 218, 310);
#    bbox = cv2.selectROI(frame, False)
    print(bbox)
    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)
    
    while True:
        
       ###############################
       # Read New frames 
       ##############################
        ok, frame = video.read()
    
        if not ok:
            break
        
        # Start timer
        timer = cv2.getTickCount()
     
        # resize the frame !
        frame = cv2.resize(frame,dim,cv2.INTER_CUBIC)
        ##########################
        # Calc Homography
        #########################
        
        # Update tracker
        ok, bbox= tracker.update(frame)
#        define_mask_with_bb(bbox)     
        
       ##############################
       ## Bounding box to MASK !
       ##############################
        mask_big_img = define_mask_with_bb(bbox)
        img2 = draw_image_matches(detector,rectified_frame,frame,None,mask_big_img)
#        img2_noMask = draw_image_matches(detector,rectified_frame,frame,None,None)    
        if(flag_compare==True):    # compare between 2 tracking
            img2_noMask = draw_image_matches(detector,rectified_frame,frame,None,None)    
            compare_image = show_compare_images(img2,img2_noMask)
        if(flag_show==True): # show both simple tracking Mosse and homography estimation
            
            # Draw bounding box
            if ok:
                # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
            else :
                # Tracking failure
                cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
     
            # Display tracker type on frame
            cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
         
            # Display FPS on frame
            cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
            MOSSE_Homography = show_compare_images(img2,frame)
           
        
        
       # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
        fps_avg  = fps_avg + fps
        count = count + 1
        
        if(flag_get_result==True):
            if(flag_compare==True):
                    out.write(compare_image)
            if(flag_show==True):
                    out.write(MOSSE_Homography)
            if(flag_show==False and flag_compare==False):
                    out.write(img2)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break
    
    
    print(["The Average FPS : ",fps_avg/count ])
    # When everything done, release the video capture and video write objects
    if(flag_get_result==True):
        out.release()
    video.release()
    cv2.destroyAllWindows()
    
#    