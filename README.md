# Image-stitching

## Abstract
Blending three images into a panorama image. <br>
Homography matrixs were calculated between left-middle, middle-right images.
Feature points found by SIFT were used to calculate matrix. The process of calculating matrix was implemented from scatch. 
In order to find robust matrix, RANSAC was used. After finding Homography matrix, backward mapping was used to prevent holes, frequently found in forward mapping.

## Implementation detail

### 1) Feature & correspondence Detection
Using SIFT for keypoint detection & descriptors. This was implemented in ```SIFT_detect``` function.<br>
After detecting keypoints, need to find correspondences between keypoints in source and reference image.<br>
The following is the details of function ```compute_raw_matches```, which computes coarse correspondences between keypoints.<br>
1) Find correspondences with descriptors, Euclidean distance was used for matching descriptors.<br>
2) Set keypoint's descriptor in source image, then calcuate Euclidean distance with all keypoint's descriptors in reference image.<br>
3) Sorting them in ascending order. The lower of the distance between descriptors, the similar with descriptors. 
4) Save best & second matching results in raw_matches.


## Results
### 1) Paris
**```Before Stitching```** <br><br>
<img src="https://user-images.githubusercontent.com/50229148/209268114-94392358-aff3-49fe-9ab3-93dbc601a328.jpg" width="300" height="200"><img src="https://user-images.githubusercontent.com/50229148/209268230-37b06945-dba9-4676-9432-cff3f68d0978.jpg" width="300" height="200"><img src="https://user-images.githubusercontent.com/50229148/209268316-78170989-36e3-4a01-8a77-9109860d5955.jpg" width="300" height="200">

**```After Stitching```** <br><br>
<img src="https://user-images.githubusercontent.com/50229148/209268013-698ef979-98a4-459f-a981-b5553a107256.jpg"><br>

### 2) Campus(1) 
**```Before Stitching```** <br><br>
<img src="https://user-images.githubusercontent.com/50229148/209268707-26f9186c-5e89-4be8-a813-d07e42800885.png" width="300" height="200"><img src="https://user-images.githubusercontent.com/50229148/209268722-a783d2b5-c8e6-4f4e-8d9d-c11e492a6fe3.png" width="300" height="200"><img src="https://user-images.githubusercontent.com/50229148/209268754-b94adbc3-d49b-4458-9a73-9fa8cfc62688.png" width="300" height="200">

**```After Stitching```** <br><br>
![result_campus2](https://user-images.githubusercontent.com/50229148/209268606-806e3559-aca5-4f2b-8c32-e210bee17ed2.jpg)

### 3) Campus(2)
**```Before Stitching```** <br><br>
<img src="https://user-images.githubusercontent.com/50229148/209268879-a980fac0-090f-4cbd-960e-b764b73f528c.png" width="300" height="200"><img src="https://user-images.githubusercontent.com/50229148/209268897-262829e4-5ed1-497c-8899-a2bdb4a70613.png" width="300" height="200"><img src="https://user-images.githubusercontent.com/50229148/209268918-c8187876-ea39-4221-a026-3eb72585dde8.png" width="300" height="200">

**```After Stitching```** <br><br>
![result_campus](https://user-images.githubusercontent.com/50229148/209268609-66156904-3699-4e20-97e7-8cfaef961631.jpg)
