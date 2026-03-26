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