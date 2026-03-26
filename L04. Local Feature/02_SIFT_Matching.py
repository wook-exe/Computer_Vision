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