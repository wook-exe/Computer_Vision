# Image Formation 실습 (E02)

### 1. Camera Calibration (`01.Calibration.py`)
<br>코드 : 01.Calibration.py

```
import cv2
import numpy as np
import glob

# 체크보드 내부 코너 개수
CHECKERBOARD = (9, 6)

# 체크보드 한 칸 실제 크기 (mm)
square_size = 25.0

# 코너 정밀화 조건
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 실제 좌표 생성
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

# 저장할 좌표
objpoints = []
imgpoints = []

images = glob.glob("c:/AIOSS/Computer_Vision/E02_Image_Formation/images/calibration_images/left*.jpg")

img_size = None

# -----------------------------
# 1. 체크보드 코너 검출
# -----------------------------
for fname in images:
    img = cv2.imread(fname)
    if img is None:
        continue
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 이미지 사이즈 저장 (모든 이미지가 동일하다고 가정)
    if img_size is None:
        img_size = gray.shape[::-1]

    # 체크보드 코너 찾기
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    # 코너를 찾았다면 정밀화(SubPix) 수행 후 배열에 저장
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        
        # (선택사항) 코너 검출 결과 화면에 출력
        # cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        # cv2.imshow('Find Corners', img)
        # cv2.waitKey(100)
        
cv2.destroyAllWindows()


# -----------------------------
# 2. 카메라 캘리브레이션
# -----------------------------
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

print("Camera Matrix K:")
print(K)

print("\nDistortion Coefficients:")
print(dist)

# -----------------------------
# 3. 왜곡 보정 시각화
# -----------------------------
if len(images) > 0:
    test_img = cv2.imread(images[0])
    h, w = test_img.shape[:2]
    
    # 최적의 새 카메라 행렬 계산
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))

    # 왜곡 보정 (Undistort)
    undistorted_img = cv2.undistort(test_img, K, dist, None, new_camera_matrix)

    # ROI에 맞춰 이미지 크롭 (검은 여백 제거)
    x, y, w_roi, h_roi = roi
    undistorted_img = undistorted_img[y:y+h_roi, x:x+w_roi]

    cv2.imshow("Original", test_img)
    cv2.imshow("Undistorted", undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```
* **핵심 코드:**
체스판의 코너를 검출하여 카메라 행렬(K)과 왜곡 계수(dist)를 구하고, 이를 바탕으로 원본 이미지의 왜곡을 보정합니다.
```python
# 1. 카메라 캘리브레이션 (카메라 행렬 및 왜곡 계수 계산)
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

# 2. 최적의 새 카메라 행렬 계산 및 왜곡 보정
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
undistorted_img = cv2.undistort(test_img, K, dist, None, new_camera_matrix)

```


* **실행 결과:**
![image]()

---

### 2. Geometric Transformation (`02.Transform.py`)

* **핵심 코드:**
이미지의 중심을 기준으로 회전 및 스케일 행렬을 만들고, 평행 이동 값을 추가하여 `warpAffine`을 적용합니다.
```python
# 1. 회전(+30도) 및 스케일(0.8배) 변환 행렬 생성
M = cv2.getRotationMatrix2D(center, angle=30, scale=0.8)

# 2. 평행 이동(+80, -40) 추가
M[0, 2] += 80
M[1, 2] -= 40

# 3. 이미지에 Affine 변환 적용
transformed_img = cv2.warpAffine(img, M, (cols, rows))

```


* **실행 결과:**
![image]()

---

### 3. Stereo Depth Estimation (`03.Depth.py`)

* **핵심 코드:**
SGBM(Semi-Global Block Matching) 알고리즘을 사용해 Disparity를 구하고, 공식($Z = \frac{f \cdot B}{d}$)을 이용해 Depth로 변환합니다.
```python
# 1. StereoSGBM을 이용한 Disparity(시차) 맵 계산
stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=80, blockSize=5, ...)
disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

# 2. Depth(거리) 맵 변환 (Z = f * B / d)
# f: 초점 거리, B: 베이스라인(카메라 간 거리)
depth_map = np.zeros_like(disparity)
valid_mask = disparity > 0
depth_map[valid_mask] = (f * B) / disparity[valid_mask]

```


* **실행 결과:**
![image]()