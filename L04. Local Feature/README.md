# L04. Local Feature - SIFT & Homography

## 📑 목차

1. [SIFT를 이용한 특징점 검출 및 시각화 (01_SIFT_keypoint.py)]
2. [SIFT를 이용한 두 영상 간 특징점 매칭 (02_SIFT_Matching.py)]
3. [호모그래피를 이용한 이미지 정합 (Image Alignment) (03_Image Alignment.py)]

## 1\. SIFT를 이용한 특징점 검출 및 시각화 (01\_SIFT\_keypoint.py)

과제 목표 : SIFT 알고리즘을 사용하여 이미지에서 특징점을 검출하고, 검출된 특징점을 원본 이미지에 시각화합니다.

### 💡 핵심 로직

이 코드는 주어진 단일 이미지(`mot_color70.jpg`)에서 SIFT 알고리즘을 사용하여 특징점을 검출하고 이를 시각화합니다. 
**특징점 추출 제한:** `cv.SIFT_create(nfeatures=...)`를 통해 특징점이 너무 많아지지 않도록 개수를 제한합니다.
**Rich Keypoints 시각화:** `cv.drawKeypoints()` 함수에 `cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS` 플래그를 적용하여 특징점의 단순한 위치뿐만 아니라 방향과 크기(Scale)까지 원 형태로 시각화합니다.
**결과 비교:** `matplotlib`을 활용해 원본 이미지와 특징점이 그려진 이미지를 나란히 출력하여 결과를 직관적으로 비교할 수 있습니다

### 💻 전체 코드

```python
import cv2 # 컴퓨터 비전 처리를 위한 OpenCV 라이브러리를 임포트합니다.
import matplotlib.pyplot as plt # 결과 이미지 시각화를 위해 matplotlib을 임포트합니다. 

# 1. 이미지 로드 및 전처리
# 원본 이미지를 BGR 포맷으로 읽어옵니다. (mot_color70.jpg 사용) 
img = cv2.imread('mot_color70.jpg')

# 이미지가 정상적으로 로드되었는지 확인합니다.
if img is None:
    print("Error: mot_color70.jpg 파일을 찾을 수 없습니다.") # 에러 메시지를 출력합니다.
    exit() # 프로그램 종료

# matplotlib 출력을 위해 BGR 색상 공간을 RGB 공간으로 변환합니다.
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 특징점 검출의 효율성을 위해 그레이스케일 이미지로 변환합니다.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. SIFT 특징점 검출
# cv.SIFT_create()를 사용하여 SIFT 객체를 생성합니다.  
# 특징점이 너무 많아지는 것을 방지하기 위해 nfeatures 값을 조정하여 최대 500개로 제한합니다. 
sift = cv2.SIFT_create(nfeatures=500)

# detectAndCompute()를 사용하여 그레이스케일 이미지에서 SIFT 특징점(keypoints)과 디스크립터(descriptors)를 검출합니다. 
keypoints, descriptors = sift.detectAndCompute(gray, None)

# 3. 특징점 시각화
# cv.drawKeypoints()를 사용하여 특징점을 이미지에 시각화합니다. 
# flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS를 설정하여 특징점의 방향과 크기(스케일)를 표시합니다. 
img_with_keypoints = cv2.drawKeypoints(img_rgb, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# 4. 결과 출력
# matplotlib을 이용하여 원본 이미지와 특징점이 시각화된 이미지를 나란히 출력하기 위한 Figure를 생성합니다. 
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# 첫 번째 서브플롯에 원본 RGB 이미지를 출력합니다.
axes[0].imshow(img_rgb)
axes[0].set_title('Original Image') # 첫 번째 이미지의 제목을 설정합니다.
axes[0].axis('off') # 축 눈금을 숨깁니다.

# 두 번째 서브플롯에 특징점이 그려진 이미지를 출력합니다.
axes[1].imshow(img_with_keypoints)
axes[1].set_title('SIFT Keypoints (Rich)') # 두 번째 이미지의 제목을 설정합니다.
axes[1].axis('off') # 축 눈금을 숨깁니다.

# 그래프 간의 간격을 자동으로 조절하여 깔끔하게 표시합니다.
plt.tight_layout()
# 화면에 최종 결과 그래프를 띄웁니다.
plt.show()
```
-----

### 🖼️ 결과
![image](https://github.com/user-attachments/assets/1ad1e7f2-d922-4833-80e7-a32d47d1a0bf)

## 2\. SIFT를 이용한 두 영상 간 특징점 매칭 (02\_SIFT\_Matching.py)

과제 목표 : SIFT 알고리즘을 사용하여 두 이미지 간의 특징점을 검출하고, 매칭된 특징점을 시각화합니다.

### 💡 핵심 로직
두 개의 이미지(mot_color70.jpg, mot_color80.jpg)에서 추출한 SIFT 특징점을 기반으로 매칭을 수행합니다.
Brute-Force Matching: cv.BFMatcher(cv.NORM_L2)를 사용하여 모든 특징점 조합 간의 거리를 계산합니다.
Ratio Test (최근접 이웃 거리 비율): 단순 매칭의 오류를 줄이기 위해 knnMatch()로 각 점당 2개의 매칭점(k=2)을 찾습니다.
첫 번째로 가까운 점과의 거리가 두 번째로 가까운 점과의 거리의 75% 미만일 때만 유효한 매칭(Good Match)으로 취급하여 정확도를 비약적으로 높입니다.

### 💻 전체 코드

```python
import cv2 # 컴퓨터 비전 처리를 위한 OpenCV 라이브러리를 임포트합니다.
import matplotlib.pyplot as plt # 결과 이미지 시각화를 위해 matplotlib을 임포트합니다. 

# 1. 이미지 로드
# cv.imread()를 사용하여 두 개의 이미지를 불러옵니다. 
img1 = cv2.imread('mot_color70.jpg') # 첫 번째 이미지 로드 
img2 = cv2.imread('mot_color83.jpg') # 두 번째 이미지 로드 

# 이미지가 정상적으로 로드되었는지 확인합니다.
if img1 is None or img2 is None:
    print("Error: 매칭할 이미지 파일을 찾을 수 없습니다.") # 에러 메시지를 출력합니다.
    exit() # 프로그램 종료

# matplotlib 시각화를 위해 두 이미지 모두 BGR에서 RGB로 색상 공간을 변환합니다.
img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# 특징점 검출을 위해 두 이미지를 그레이스케일로 변환합니다.
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# 2. SIFT 특징점 및 디스크립터 추출
# cv.SIFT_create()를 사용하여 SIFT 객체를 생성합니다. 
sift = cv2.SIFT_create()

# 첫 번째 이미지에서 특징점과 디스크립터를 추출합니다.
kp1, des1 = sift.detectAndCompute(gray1, None)
# 두 번째 이미지에서 특징점과 디스크립터를 추출합니다.
kp2, des2 = sift.detectAndCompute(gray2, None)

# 3. 특징점 매칭
# cv.BFMatcher()를 사용하여 두 영상 간 특징점을 매칭할 객체를 생성합니다. L2 노름을 사용합니다. 
bf = cv2.BFMatcher(cv2.NORM_L2)

# knnMatch()를 사용하여 각 특징점당 가장 유사한 2개의 이웃(k=2) 매칭점을 찾습니다. 
matches = bf.knnMatch(des1, des2, k=2)

# 좋은 매칭점만을 선별하기 위한 빈 리스트를 생성합니다.
good_matches = []
# knnMatch로 반환된 매칭점들 중 첫 번째(m)와 두 번째(n) 매칭 거리를 순회합니다.
for m, n in matches:
    # 최근접 이웃 거리 비율을 적용하여 매칭 정확도를 높입니다. (Lowe's Ratio Test, 임계값 0.75) 
    if m.distance < 0.75 * n.distance:
        good_matches.append([m]) # 조건을 만족하는 DMatch 객체만 리스트에 추가합니다. 

# 4. 결과 시각화
# cv.drawMatches() (또는 knn 매칭용 drawMatchesKnn)를 사용하여 매칭 결과를 시각화합니다. 
img_matches = cv2.drawMatchesKnn(img1_rgb, kp1, img2_rgb, kp2, good_matches, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

# matplotlib을 이용하여 매칭 결과를 출력하기 위해 Figure를 생성합니다. 
plt.figure(figsize=(15, 8))
# 시각화할 이미지를 화면에 그립니다.
plt.imshow(img_matches)
# 이미지 제목을 설정합니다.
plt.title('SIFT Feature Matching (KNN & Ratio Test)')
# 축 눈금을 숨겨 깔끔하게 보여줍니다.
plt.axis('off')
# 화면에 그래프를 띄웁니다.
plt.show()
```

-----

### 🖼️ 결과
![image](https://github.com/user-attachments/assets/807b5d44-cbb4-439e-ad1d-4be76bab2cd3)

## 3\. 호모그래피를 이용한 이미지 정합 (Image Alignment) (03\_Image Alignment.py)

과제 목표 : SIFT 특징점과 호모그래피를 이용하여 두 이미지를 정합(Alignment)하고, 결과를 시각화합니다.

### 💡 핵심 로직
추출된 SIFT 대응점을 바탕으로 두 이미지의 시점을 맞추는 투영 변환(호모그래피)을 수행하여 파노라마를 생성합니다.
Homography 및 RANSAC: cv.findHomography() 함수를 활용하여 3x3 변환 행렬을 찾습니다.
이 과정에서 cv.RANSAC 알고리즘을 사용하여 오매칭된 이상점(Outlier)의 영향을 배제하고 Inlier 데이터만으로 강건한 행렬을 계산합니다.
Warp Perspective: 구해진 행렬을 바탕으로 cv.warpPerspective()를 실행하여 한 이미지를 다른 이미지의 시점 평면으로 변환(정렬)시킵니다.
이때 출력 캔버스의 크기는 두 이미지를 합친 파노라마 크기 (w1+w2, max(h1,h2))로 설정하여 영상이 잘리지 않도록 합니다.

### 💻 전체 코드

```python
import cv2 # 컴퓨터 비전 처리를 위한 OpenCV 라이브러리를 임포트합니다.
import numpy as np # 행렬 및 배열 연산을 위해 NumPy 라이브러리를 임포트합니다.
import matplotlib.pyplot as plt # 결과 이미지 시각화를 위해 matplotlib을 임포트합니다.

# 1. 이미지 로드
# cv.imread()를 사용하여 두 개의 샘플 이미지를 불러옵니다. (img1.jpg, img2.jpg 선택)
img1 = cv2.imread('img1.jpg')
img2 = cv2.imread('img2.jpg')

# 파일이 없는 경우를 대비한 예외 처리입니다.
if img1 is None or img2 is None:
    print("Error: 정합할 img1.jpg 또는 img2.jpg 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    exit()

# SIFT 계산을 위해 그레이스케일로 변환합니다.
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# 2. SIFT 특징점 검출
# cv.SIFT_create()를 사용하여 SIFT 객체를 생성합니다. 
sift = cv2.SIFT_create()

# 두 이미지에서 특징점과 디스크립터를 각각 추출합니다.
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# 3. 특징점 매칭 및 필터링
# cv.BFMatcher()를 생성합니다. 
bf = cv2.BFMatcher(cv2.NORM_L2)
# knnMatch()를 사용하여 특징점을 매칭합니다. (k=2) 
matches = bf.knnMatch(des1, des2, k=2)

# 좋은 매칭점만 선별하기 위한 리스트입니다. 
good = []
# 거리 비율이 임계값(0.7) 미만인 매칭점만 선별합니다. 
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m) # 통과된 매칭 객체를 저장합니다.

# 4. 호모그래피 계산 및 이미지 정합
# 호모그래피를 계산하려면 최소 4개 이상의 매칭점이 필요합니다.
if len(good) > 4:
    # 매칭된 특징점들의 좌표를 img2(출발지, src)와 img1(목적지, dst)에서 각각 추출하여 float32 배열로 변환합니다.
    src_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    
    # cv.findHomography()를 사용하여 호모그래피 행렬(M)을 계산합니다. 
    # cv.RANSAC을 사용하여 이상점(Outlier) 영향을 줄입니다.  허용 오차는 5.0 픽셀입니다.
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    # 두 이미지를 합친 파노라마 크기(w1+w2, max(h1,h2))를 설정하기 위해 원본 이미지의 크기를 구합니다. 
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    pano_w = w1 + w2 # 결과 이미지의 너비
    pano_h = max(h1, h2) # 결과 이미지의 높이
    
    # cv.warpPerspective()를 사용하여 한 이미지(img2)를 변환하여 정렬합니다. 
    # 출력 크기를 앞서 구한 파노라마 크기로 설정합니다. 
    warped_img = cv2.warpPerspective(img2, M, (pano_w, pano_h))
    
    # 변환된 캔버스의 좌측 영역에 원본 이미지 1(img1)을 덮어씌워 하나로 정합합니다.
    warped_img[0:h1, 0:w1] = img1
    
    # 시각화를 위해 BGR 색상을 RGB로 변환합니다.
    warped_img_rgb = cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB)
    
    # 5. 결과 시각화
    # Inlier 매칭점만 표시하기 위한 파라미터입니다.
    draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=mask.ravel().tolist(), flags=2)
    # 특징점 매칭 결과를 그립니다.
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    img_matches_rgb = cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB)
    
    # 변환된 이미지와 특징점 매칭 결과를 나란히(2행 1열 구조) 출력하기 위한 Figure를 생성합니다. 
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 상단에 매칭 결과 출력
    axes[0].imshow(img_matches_rgb)
    axes[0].set_title('Matching Result (Inliers Only)')
    axes[0].axis('off')
    
    # 하단에 정합된(Warped) 이미지 출력
    axes[1].imshow(warped_img_rgb)
    axes[1].set_title('Warped and Aligned Image (Homography)')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
else:
    # 매칭점이 부족할 경우 에러를 출력합니다.
    print("Not enough matches are found - {}/{}".format(len(good), 4))
```

-----

### 🖼️ 결과
![image](https://github.com/user-attachments/assets/16826f35-dca5-4134-bd9c-5156734e1cc0)