# Introduction
Homography Estimation &amp; Tracking is a python project that have two main goals track objects using correlation filter and to find the orientation using RANSAC-like methods.  
There is two main approach to finding solutions :  
1- reference image approach: in this method, we create a perspective correction to first frame only and perform features extraction, then for every frame later we compare/match between the current frame and the reference one.  
2 - two successive images: the feature extraction done for the incoming frame  and matching done between 2 successive frames in specific regions 

#  How to Use  :D 
Firstly the user needs to pick 4 points that surround the desired object, then surround the regions of interest (ROI) with a bounding box, in this script, it's done manually since we need to check the perfect fitting to our ROI  in further steps this part will be done automatically as object detections.
# Video 
 The link of the video used in the code   
 [Video1](https://drive.google.com/open?id=174KgRP8HQ9aldRMzzC_MEgL7cNUnxRdo )

 

#  Flag Guide 
For each script, there are flags that give some utilities, (i.e compare between results, show the key points, the bounding box of MOSSE tracking and others) 

