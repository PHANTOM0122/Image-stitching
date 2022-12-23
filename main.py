import cv2
from numpy import linalg
import numpy as np
import math

# RANSAC에서 동일한 결과를 얻기 위한 Seed 고정
np.random.seed(1)

# SIFT 알고리즘을 이용하여 특징점들과 기술자들을 추출하는 함수
def SIFT_detect(img):
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    kp = np.float32([kp.pt for kp in kp])
    return kp, des

# 기술자들을 정규화 시켜주는 함수
def descriptor_normalizer(feature_vec):
  # normalize descriptors to zero mean and unit std
  m = np.expand_dims(feature_vec.mean(axis=1), axis=1)
  s = np.expand_dims(feature_vec.std(axis=1), axis=1)
  return (feature_vec-m)/s

# 특징점들의 기술자들중 유사한 것끼리 매칭을 진행해준다
def compute_raw_matches(des1, des2, numOfmatches = 2):
    raw_matches = []
    for i, source in enumerate(des1):
        distance_list = np.array([linalg.norm(source - des2, axis=1), # 두 기술자 벡터의 유클리디안 거리(L2 norm)를 구한다
                                  i * np.ones(len(des2)), [i for i in range(len(des2))]]).transpose()
        sd = distance_list[distance_list[:, 0].argsort()] # 거리에 따라 정렬시킨다
        raw_matches.append(sd[:numOfmatches])
    raw_matches = np.array(raw_matches)
    return raw_matches

# Raw한 매칭 중에서 좋은 매칭들을 찾아주는 함수
def filter_matches(raw_matches, distance_ratio = 1.0):
    good_matches = []
    for match in raw_matches:
        # 가장 좋은 match(match[0.0])의 거리 오차가 그 다음 match(match[1,0])보다 distance ratio의 비율보다 작을 것을 good match로 판단한다.
        if len(match) == 2 and match[0,0] < match[1,0] * distance_ratio:
            good_matches.append((match[0,1], match[0,2]))

    return good_matches

# Homography matrix를 계산해주는 함수이다
def calculate_homography(pts1, pts2):
    homography_list = []

    # 두 이미지의 특징점들을 이용하여 Homography를 계산한다.
    for a, b in zip(pts1, pts2):
        p1 = np.array([a[0], a[1], 1]) # 특징점 1의 homogeneous 좌표
        p2 = np.array([b[0], b[1], 1]) # 특징점 2의 homogeneous 좌표

        a2 = [0, 0, 0, -p2[2] * p1[0], -p2[2] * p1[1], -p2[2] * p1[2],
              p2[1] * p1[0], p2[1] * p1[1], p2[1] * p1[2]]

        a1 = [-p2[2] * p1[0], -p2[2] * p1[1], -p2[2] * p1[2], 0, 0, 0,
              p2[0] * p1[0], p2[0] * p1[1], p2[0] * p1[2]]

        homography_list.append(a1)
        homography_list.append(a2)

    matrixA = np.array(homography_list)

    # svd composition
    u, s, v = np.linalg.svd(matrixA)

    # reshape the min singular value into a 3 by 3 matrix
    h = np.reshape(v[8, :], (3, 3))

    # normalize and now we ve h
    h = (1 / h[2, 2]) * h
    return h

# 두 점간의 거리를 구해주는 함수이다
def geometricDistance(cpa, cpb, h):
    p1 = np.transpose(np.array([cpa[0], cpa[1], 1])) # Homogeneous 좌표계로 만든다
    estimatep2 = np.dot(h, p1) # p1을 Homography matrix와 내적하여 변환시킨다
    estimatep2 = (1/estimatep2.item(2))*estimatep2 # 변환된 p1을 (x, y, 1) 꼴로 만들어주기 위해서, 마지막 2번째 element으로 나눠준다  

    p2 = np.transpose(np.array([cpb[0], cpb[1], 1])) # p2를 Homogeneous 좌표계로 만든다
    error = p2 - estimatep2 # 변환된 p1과 p2간의 차이 벡터를 구한다 
    return linalg.norm(error) # 차이 벡터의 L2 norm을 반환한다

# RANSAC을 이용하여 Homography matrix를 계산
def match_ransac(pts1, pts2, threshold):
    maxInliers = 0
    homography = None
    for i in range(1000):
        # homography 계산을 위해 4개의 random point 찾기
        index_rand = np.random.randint(0, len(pts1), 4) # 4개의 임의의 인덱스 추출
        pts1_rand = pts1[index_rand]
        pts2_rand = pts2[index_rand]

        # Random 포인트를 기반으로 homography matrix 계산
        H = calculate_homography(pts1_rand, pts2_rand)

        status = np.zeros(pts1.shape[0])
        inliers = 0 # Inlier의 갯수를 0으로 초기화

        for i in range(len(pts1)):
            distance = geometricDistance(pts1[i], pts2[i], H) # Homography matrix에 의해 변환된 점과 대응되는 점 간의 거리 구하기
            if distance < threshold: # Threshold 이하이면 inlier로 판단
                status[i] = 1 # inlier 인덱스 체크
                inliers += 1 # inlier의 갯수 증가
        
        # 가장 많은 inlier를 가지고 있는 상태를 저장한다
        if inliers > maxInliers:
            maxInliers = inliers
            homography = H
            
        # inlier 갯수가 최대치에 도달하면 최종 homogrphy 구한다
        if maxInliers > (len(pts1) * 0.5):
            break

    return homography, status

# 2개의 이미지를 stitch 해주는 함수, 위에서 정의한 함수들이 사용된다
def images_stitching(imgL, imgR):

    # Extract SIFT keypoints and descriptors (각 이미지에 대해 keypoint와 descriptor를 계산한다)
    kpL, desL = SIFT_detect(imgL)
    kpR, desR = SIFT_detect(imgR)

    # Normalize descriptors (Descriptor 정규화를 진행)
    normalized_desL = descriptor_normalizer(desL)
    normalized_desR = descriptor_normalizer(desR)

    # 먼저 초기의 Raw matches를 구한다
    raw_matches = compute_raw_matches(normalized_desL, normalized_desR)

    # Refine하여 Good matches를 구한다
    matches = filter_matches(raw_matches, distance_ratio=0.6)

    if len(matches) > 4:
        # construct the two sets of points
        ptsL = np.float32([kpL[int(i)] for (i, _) in matches]) # 특징점 1
        ptsR = np.float32([kpR[int(i)] for (_, i) in matches]) # 특징점 2

        # RANSAC을 이용해 두 점들의 집합간 Homography matrix를 구한다
        H, status = match_ransac(ptsL, ptsR, 5.0)

        # Homography 행렬을 반환한다
        return matches, H, status, kpL, kpR

    # Homography가 계산이 안되는 경우 아무것도 반환하지 않는다
    return None

# ( x, y, z )꼴의 homogeneous 좌표를 (x,y,1)로 만들고 x,y점을 반환하는 함수
def homogeneous_coordinate(coordinate):
    x = coordinate[0]/coordinate[2]
    y = coordinate[1]/coordinate[2]
    return x, y

# Backward image warping
def warp(image, homography):
    print("Warping is started.")
    image_array = np.array(image)
    row_number, column_number = int(image_array.shape[0]), int(image_array.shape[1])

    # 사진 네 모서리 꼭지점을 homogenous 좌표계로 바꾼다
    up_left_cor_x, up_left_cor_y = homogeneous_coordinate(np.dot(homography, [[0],[0],[1]]))
    up_right_cor_x, up_right_cor_y = homogeneous_coordinate(np.dot(homography, [[column_number-1],[0],[1]]))
    low_left_cor_x, low_left_cor_y = homogeneous_coordinate(np.dot(homography, [[0],[row_number-1],[1]]))
    low_right_cor_x, low_right_cor_y = homogeneous_coordinate(np.dot(homography, [[column_number-1],[row_number-1],[1]]))

    # homography에 의해 변환된 네 꼭지점들
    x_values = [up_left_cor_x, up_right_cor_x, low_right_cor_x, low_left_cor_x]
    y_values = [up_left_cor_y, up_right_cor_y, low_left_cor_y,  low_right_cor_y]
    print("x_values: ", x_values, "\n y_values: ", y_values)

    # 변환된 꼭지점들의 최소 x, y 값
    offset_x = math.floor(min(x_values))
    offset_y = math.floor(min(y_values))
    print("offset_x: ", offset_x, "\t size_y: ", offset_y)

    # 변환된 꼭지점들의 최대 x, y 값
    max_x = math.ceil(max(x_values))
    max_y = math.ceil(max(y_values))
    
    # output 사이즈: 결과적으로 x_values, y_values의 최대와 최소의 차이를 사이즈로 만든다
    size_x = max_x - offset_x 
    size_y = max_y - offset_y
    print("size_x: ", size_x, "\t size_y: ", size_y)

    # Backward Warping을 위해서 homography 역행렬을 구한다
    homography_inverse = np.linalg.inv(homography)
    print("Homography inverse: ", "\n", homography_inverse)
    
    # ouput을 먼저 만든다
    result = np.zeros((size_y, size_x, 3))
    
    # Backward warping은 output의 좌표에 대응되는 값을 source에서 찾는다
    # 이로 인해, output에서는 black hole들이 생기지 않는다는 장점이 있다
    for x in range(size_x): # ouput의 x좌표
        for y in range(size_y): # ouput의 y좌표
            # Inverse Homography matrix를 이용하여 
            point_xy = homogeneous_coordinate(np.dot(homography_inverse, [[x+ offset_x], [y+offset_y], [1]]))
            point_x = int(point_xy[0]) # source의 x좌표
            point_y = int(point_xy[1]) # source의 y좌표
            if (point_x >= 0 and point_x < column_number and point_y >= 0 and point_y < row_number): # output 영역 안에서만 정의
                result[y, x, :] = image_array[point_y, point_x, :]

    print("Warping is completed.")
    return result, offset_x, offset_y

# 3개의 이미지를 하나의 이미지로 만들어준다
def blend3images(left, middle, right, left_middle_offset_x, left_middle_offset_y, right_middle_offset_x, right_middle_offset_y):

    # 각 이미지들의 높이와 너비를 파악
    rows_left, columns_left = int(left.shape[0]), int(left.shape[1])
    rows_middle, columns_middle = int(middle.shape[0]), int(middle.shape[1])
    rows_right, columns_right = int(right.shape[0]), int(right.shape[1])

    # x좌표의 최대, 최소를 구한다
    x_min = min([left_middle_offset_x, right_middle_offset_x, 0])
    x_max = max([left_middle_offset_x+columns_left, right_middle_offset_x+columns_right, columns_middle])
    
    # y좌표의 최대와 최소를 구한다
    y_min = min([left_middle_offset_y, right_middle_offset_y, 0])
    y_max = max([rows_left+left_middle_offset_y, rows_right+right_middle_offset_y, rows_middle])
    
    # Output의 사이즈를 결정
    size_x = x_max - x_min
    size_y = y_max - y_min

    # 3개의 이미지가 섞인 출력을 먼저 정의
    blending = np.zeros((size_y, size_x, 3))

    # left 이미지부터 출력에 붙여준다
    blending[:rows_left, :columns_left, :] = left[:, :, :]

    # right 이미지를 출력에 붙여준다
    blending[size_y-rows_right:, size_x-columns_right:, :] = right[:, :, :]
    blending[size_y - rows_right:, size_x - columns_right:, :] = np.where(
        blending[size_y - rows_right:, size_x - columns_right:, :] == [0, 0, 0],
        right[:, :, :], blending[size_y - rows_right:, size_x - columns_right:, :])

    # middle 이미지를 출력에 붙여준다
    blending[-left_middle_offset_y:rows_middle - left_middle_offset_y,
    -left_middle_offset_x:columns_middle - left_middle_offset_x, :] = middle[:, :, :]
    
    # output의 크기 이외의 영역들을 잘라준다
    blending[-left_middle_offset_y:rows_middle-left_middle_offset_y, -left_middle_offset_x:columns_middle-left_middle_offset_x, :] = \
       np.where(np.mean(middle[:2], axis=0) <
                np.mean(blending[-left_middle_offset_y:rows_middle-left_middle_offset_y, -left_middle_offset_x:columns_middle-left_middle_offset_x, :][:2], axis=0),
                blending[-left_middle_offset_y:rows_middle-left_middle_offset_y, -left_middle_offset_x:columns_middle-left_middle_offset_x, :], middle)

    print("Blending is completed.")
    return blending

if __name__ == '__main__':

    imgL = cv2.imread('left_school.png') # 좌측 이미지
    imgM = cv2.imread('middle_school.png') # 중간 이미지
    imgR = cv2.imread('right_school.png') # 오른쪽 이미지

    # matches features between two images
    M1 = images_stitching(imgL, imgM) # 좌측-중간 이미지 매칭
    M2 = images_stitching(imgR, imgM) # 중간-오른쪽 이미지 매칭

    if M1 is None or M2 is None:
        print("There's no matches between two images!")

    # otherwise, apply a perspective warp to stitch the images together
    (matches1, H1, status, kpsA, kpsB) = M1 # 매칭 정보로 부터 Homography matrix를 도출
    (matches2, H2, status2, kpsA2, kpsB2) = M2 # 매칭 정보로 부터 Homography matrix를 도출

    # Backward warping1
    warped_image_source1, source1_offset_x, source1_offset_y = warp(imgL, H1) # Backward warping을 이용하여 이미지 변환
    warped_image_source2, source2_offset_x, source2_offset_y = warp(imgR, H2) # Backward warping을 이용하여 이미지 변환
    
    # 3장의 이미지를 하나의 파노라마 이미지로 만들게 됩니다
    final_image = blend3images(warped_image_source1, np.array(imgM), warped_image_source2,
                                  source1_offset_x, source1_offset_y, source2_offset_x, source2_offset_y)
    
    # OpenCV를 이용한 이미지 시각화 및 저장
    cv2.imshow('final_image', np.asarray(final_image, dtype=np.uint8))
    cv2.waitKey(0)
    # cv2.imwrite('result_campus.jpg', np.asarray(final_image, dtype=np.uint8)) # 이미지 저장
