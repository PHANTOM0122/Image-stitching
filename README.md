# Image-stitching

## Abstract
Blending three images into a panorama image. <br>
Homography matrixs were calculated between left-middle, middle-right images.
Feature points found by SIFT were used to calculate matrix. The process of calculating matrix was implemented from scatch. 
In order to find robust matrix, RANSAC was used. After finding Homography matrix, backward mapping was used to prevent holes, frequently found in forward mapping.

## Results
1) Paris

2) Campus(1)
3) Campus(2)
