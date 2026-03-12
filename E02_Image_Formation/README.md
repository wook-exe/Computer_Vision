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
![image](https://github.com/user-attachments/assets/ed2a86b8-700b-456e-975d-9aca12e418ce)

---

### 2. Geometric Transformation (`02.Transform.py`)
<br>코드 : 02.Transform.py

```
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread("c:/AIOSS/Computer_Vision/E02_Image_Formation/images/rose.png")
if img is None:
    raise FileNotFoundError("rose.png 이미지를 찾지 못했습니다.")

rows, cols = img.shape[:2]

# 조건
angle = 30          # +30도 회전
scale = 0.8         # 크기 0.8배
tx = 80             # x축 +40px
ty = -40            # y축 -20px

# 이미지 중심
center = (cols / 2, rows / 2)

# 회전 + 스케일 행렬 생성
M = cv.getRotationMatrix2D(center, angle, scale)

# 평행이동 추가
M[0, 2] += tx
M[1, 2] += ty

# 원본 크기 유지
transformed_img = cv.warpAffine(
    img,
    M,
    (cols, rows),
    flags=cv.INTER_LINEAR,
    borderMode=cv.BORDER_CONSTANT,
    borderValue=(0, 0, 0)
)

# 시각화
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv.cvtColor(transformed_img, cv.COLOR_BGR2RGB))
plt.title("Rotated + Scaled + Translated")
plt.axis("off")

plt.show()
```
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
![image](https://github.com/user-attachments/assets/a0f64c53-5ba4-49fa-bd07-311f827833f2)

---

### 3. Stereo Depth Estimation (`03.Depth.py`)
<br>코드 : 03.Depth.py

```
import cv2
import numpy as np
from pathlib import Path

# 출력 폴더 생성
output_dir = Path("./outputs")
output_dir.mkdir(parents=True, exist_ok=True)

# 좌/우 이미지 불러오기
left_color = cv2.imread("c:/AIOSS/Computer_Vision/E02_Image_Formation/images/left.png")
right_color = cv2.imread("c:/AIOSS/Computer_Vision/E02_Image_Formation/images/right.png")

if left_color is None or right_color is None:
    raise FileNotFoundError("좌/우 이미지를 찾지 못했습니다.")


# 카메라 파라미터
f = 700.0
B = 0.12

# ROI 설정
rois = {
    "Painting": (55, 50, 130, 110),
    "Frog": (90, 265, 230, 95),
    "Teddy": (310, 35, 115, 90)
}

# 그레이스케일 변환
left_gray = cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY)

# -----------------------------
# 1. Disparity 계산
# -----------------------------
window_size = 5
min_disp = 0
num_disp = 16 * 5 # 반드시 16의 배수여야 함

stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=window_size,
    P1=8 * 3 * window_size**2,
    P2=32 * 3 * window_size**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)
# disparity 계산 (OpenCV는 결과를 16배 곱해서 반환하므로 16.0으로 나눔)
disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

# -----------------------------
# 2. Depth 계산
# Z = fB / d
# -----------------------------
depth_map = np.zeros_like(disparity)
valid_mask = disparity > 0  # disparity가 0 이하인 부분은 유효하지 않음

# 유효한 disparity 픽셀에 대해서만 depth 계산
depth_map[valid_mask] = (f * B) / disparity[valid_mask]

# -----------------------------
# 3. ROI별 평균 disparity / depth 계산
# -----------------------------
results = {}

for name, (x, y, w, h) in rois.items():
    # ROI 영역 추출
    roi_disp = disparity[y:y+h, x:x+w]
    roi_depth = depth_map[y:y+h, x:x+w]

    # 해당 ROI 내에서 유효한 픽셀 마스크
    roi_valid_mask = roi_disp > 0

    if np.any(roi_valid_mask):
        mean_disp = np.mean(roi_disp[roi_valid_mask])
        mean_depth = np.mean(roi_depth[roi_valid_mask])
    else:
        mean_disp = 0.0
        mean_depth = 0.0

    results[name] = {"mean_disp": mean_disp, "mean_depth": mean_depth}

# -----------------------------
# 4. 결과 출력
# -----------------------------
print(f"{'ROI Name':<10} | {'Mean Disparity (px)':<20} | {'Mean Depth (m)':<15}")
print("-" * 55)
for name, res in results.items():
    print(f"{name:<10} | {res['mean_disp']:<20.4f} | {res['mean_depth']:<15.4f}")

# -----------------------------
# 5. disparity 시각화
# 가까울수록 빨강 / 멀수록 파랑
# -----------------------------
disp_tmp = disparity.copy()
disp_tmp[disp_tmp <= 0] = np.nan

if np.all(np.isnan(disp_tmp)):
    raise ValueError("유효한 disparity 값이 없습니다.")

d_min = np.nanpercentile(disp_tmp, 5)
d_max = np.nanpercentile(disp_tmp, 95)

if d_max <= d_min:
    d_max = d_min + 1e-6

disp_scaled = (disp_tmp - d_min) / (d_max - d_min)
disp_scaled = np.clip(disp_scaled, 0, 1)

disp_vis = np.zeros_like(disparity, dtype=np.uint8)
valid_disp = ~np.isnan(disp_tmp)
disp_vis[valid_disp] = (disp_scaled[valid_disp] * 255).astype(np.uint8)

disparity_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

# -----------------------------
# 6. depth 시각화
# 가까울수록 빨강 / 멀수록 파랑
# -----------------------------
depth_vis = np.zeros_like(depth_map, dtype=np.uint8)

if np.any(valid_mask):
    depth_valid = depth_map[valid_mask]

    z_min = np.percentile(depth_valid, 5)
    z_max = np.percentile(depth_valid, 95)

    if z_max <= z_min:
        z_max = z_min + 1e-6

    depth_scaled = (depth_map - z_min) / (z_max - z_min)
    depth_scaled = np.clip(depth_scaled, 0, 1)

    # depth는 클수록 멀기 때문에 반전
    depth_scaled = 1.0 - depth_scaled
    depth_vis[valid_mask] = (depth_scaled[valid_mask] * 255).astype(np.uint8)

depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

# -----------------------------
# 7. Left / Right 이미지에 ROI 표시
# -----------------------------
left_vis = left_color.copy()
right_vis = right_color.copy()

for name, (x, y, w, h) in rois.items():
    cv2.rectangle(left_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(left_vis, name, (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.rectangle(right_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(right_vis, name, (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# -----------------------------
# 8. 저장
# -----------------------------
cv2.imwrite(str(output_dir / "disparity_color.png"), disparity_color)
cv2.imwrite(str(output_dir / "depth_color.png"), depth_color)
cv2.imwrite(str(output_dir / "left_rois.png"), left_vis)
cv2.imwrite(str(output_dir / "right_rois.png"), right_vis)

# -----------------------------
# 9. 출력
# -----------------------------
cv2.imshow("Disparity Heatmap", disparity_color)
cv2.imshow("Depth Heatmap", depth_color)
cv2.imshow("Left ROI", left_vis)
cv2.imshow("Right ROI", right_vis)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

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
![image](https://github.com/user-attachments/assets/e2770b62-1617-4436-8552-f875725cdb23)