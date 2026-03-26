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