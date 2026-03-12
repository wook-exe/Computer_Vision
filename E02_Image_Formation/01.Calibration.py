import cv2 # OpenCV 라이브러리를 가져옴. 이미지 및 영상 처리를 위해 사용됨.
import numpy as np # 수치 계산 및 배열(행렬) 연산을 위한 NumPy 라이브러리를 가져옴.
import glob # 특정 패턴과 일치하는 파일 경로를 쉽게 찾기 위해 glob 라이브러리를 가져옴.

# 체크보드 내부 코너 개수
CHECKERBOARD = (9, 6) # 체스판 이미지에서 찾고자 하는 내부 코너의 교차점 개수(가로 9개, 세로 6개)를 튜플로 정의함.

# 체크보드 한 칸 실제 크기 (mm)
square_size = 25.0 # 체스판의 검은색/흰색 사각형 한 칸의 실제 물리적 크기(여기서는 25mm)를 정의함.

# 코너 정밀화 조건
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # 코너 위치를 픽셀 단위보다 더 정밀하게(SubPixel) 찾기 위한 반복 알고리즘의 종료 조건(최대 30번 반복 또는 오차 0.001 이하)을 설정함.

# 실제 좌표 생성
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32) # 체스판 코너들의 3차원 실제 공간 좌표(X, Y, Z)를 저장할 (54, 3) 크기의 0으로 채워진 빈 배열을 생성함. (Z는 모두 0으로 가정)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) # X, Y 좌표를 0부터 8, 0부터 5까지의 격자(grid) 형태로 자동 생성하여 배열에 채워 넣음.
objp *= square_size # 생성된 격자 좌표에 실제 사각형 크기(25.0)를 곱하여 실제 물리적 거리(mm) 단위로 스케일을 맞춤.

# 저장할 좌표
objpoints = [] # 여러 장의 캘리브레이션 이미지에서 구한 3차원 실제 공간의 코너 좌표들을 모아둘 빈 리스트를 생성함.
imgpoints = [] # 여러 장의 캘리브레이션 이미지에서 실제로 검출된 2차원 이미지 평면상의 코너 픽셀 좌표들을 모아둘 빈 리스트를 생성함.

images = glob.glob("c:/AIOSS/Computer_Vision/E02_Image_Formation/images/calibration_images/left*.jpg") # 지정된 폴더 경로 안에서 이름이 'left'로 시작하고 확장자가 '.jpg'인 모든 이미지 파일의 경로를 찾아 리스트로 만듦.

img_size = None # 나중에 이미지의 가로세로 해상도(크기)를 추출하여 저장하기 위해 변수를 초기화해 둠.

# -----------------------------
# 1. 체크보드 코너 검출
# -----------------------------
for fname in images: # glob으로 찾은 이미지 파일 경로 리스트를 하나씩 순회하며 반복문을 실행함.
    img = cv2.imread(fname) # 해당 파일 경로(fname)로부터 이미지를 읽어와서 변수 img에 저장함.
    if img is None: # 만약 경로가 잘못되었거나 파일이 깨져서 이미지를 정상적으로 불러오지 못했다면
        continue # 이번 이미지는 코너 검출을 건너뛰고 바로 다음 파일로 넘어감.
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # OpenCV의 컬러 이미지를 그레이스케일(흑백) 이미지로 변환함. (코너 검출 알고리즘은 흑백에서 더 정확하고 빠름)
    
    # 이미지 사이즈 저장 (모든 이미지가 동일하다고 가정)
    if img_size is None: # 아직 이미지 크기가 저장되지 않은 반복문 첫 번째 바퀴라면
        img_size = gray.shape[::-1] # 흑백 이미지의 형태(세로, 가로) 배열을 거꾸로 뒤집어 (가로, 세로) 튜플 형태로 img_size 변수에 저장함.

    # 체크보드 코너 찾기
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None) # 흑백 이미지 속에서 지정한 크기(9, 6)의 체스판 내부 코너를 찾음. ret은 성공 여부, corners는 찾은 픽셀 좌표들임.

    # 코너를 찾았다면 정밀화(SubPix) 수행 후 배열에 저장
    if ret == True: # 만약 사진 안에서 체스판 코너를 성공적으로 모두 검출했다면
        objpoints.append(objp) # 현재 이미지에 대응하는 3차원 실제 물리 공간 좌표 원본(objp)을 objpoints 리스트에 추가함.
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria) # 대략적으로 찾은 픽셀 코너 좌표를 알고리즘을 통해 소수점 단위(SubPixel) 위치까지 더 정밀하게 보정함.
        imgpoints.append(corners2) # 정밀하게 보정된 2차원 이미지 코너 좌표 그룹을 imgpoints 리스트에 차곡차곡 추가함.
        
        # (선택사항) 코너 검출 결과 화면에 출력
        # cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret) # 원본 컬러 이미지 위에 검출한 코너점들을 알록달록한 선으로 이어 그림.
        # cv2.imshow('Find Corners', img) # 코너가 그려진 이미지를 'Find Corners'라는 이름의 팝업 창에 띄워 시각적으로 확인시켜줌.
        # cv2.waitKey(100) # 100밀리초(0.1초) 동안 대기하며 화면을 갱신함. (여러 사진이 연속으로 지나가는 것을 보기 위함)
        
cv2.destroyAllWindows() # 검출 반복문이 완전히 끝난 후, 코너 검출을 보여주기 위해 열려있던 모든 OpenCV 이미지 팝업 창을 닫음.

# -----------------------------
# 2. 카메라 캘리브레이션
# -----------------------------
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None) # 누적된 3차원 실제 좌표들과 2차원 이미지 좌표 쌍들을 이용해, 카메라 렌즈의 왜곡 계수(dist)와 내부 행렬(K) 파라미터를 역산하여 계산함.

print("Camera Matrix K:") # 카메라 내부 파라미터 행렬 K를 출력하겠다는 안내 문구를 콘솔 창에 출력함.
print(K) # 초점 거리와 렌즈 광학 중심점 등의 정보가 담긴 3x3 카메라 매트릭스 계산 결과를 출력함.

print("\nDistortion Coefficients:") # 렌즈의 왜곡 계수 배열을 출력하겠다는 안내 문구를 줄바꿈하여 출력함.
print(dist) # 카메라 렌즈의 방사형(가장자리가 휘어짐) 및 접선형(비뚤어짐) 왜곡 정도를 수치화한 1차원 배열 값을 출력함.

# -----------------------------
# 3. 왜곡 보정 시각화
# -----------------------------
if len(images) > 0: # 사진을 1장이라도 성공적으로 불러와서 검출 과정을 마쳤다면 왜곡 보정(Undistortion) 테스트를 진행함.
    test_img = cv2.imread(images[0]) # 보정 테스트를 위해 불러왔던 리스트의 맨 첫 번째 사진을 원본 이미지로 다시 읽어옴.
    h, w = test_img.shape[:2] # 방금 불러온 테스트 원본 이미지의 높이(h)와 너비(w) 픽셀 크기 정보를 추출함.
    
    # 최적의 새 카메라 행렬 계산
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h)) # 앞서 계산된 행렬(K)과 왜곡 계수(dist)를 바탕으로, 왜곡을 평평하게 펼 때 이미지가 잘려나가는 것을 최소화(alpha=1)해주는 새로운 최적 행렬과 보존 영역(roi)을 계산함.

    # 왜곡 보정 (Undistort)
    undistorted_img = cv2.undistort(test_img, K, dist, None, new_camera_matrix) # 원본 이미지 픽셀들에 카메라 행렬과 왜곡 계수 역연산을 적용하여, 볼록/오목한 렌즈 왜곡이 일직선으로 펴진 새로운 이미지를 만듦.

    # ROI에 맞춰 이미지 크롭 (검은 여백 제거)
    x, y, w_roi, h_roi = roi # 최적 행렬 계산 단계에서 함께 반환받은 유효 이미지 영역(roi)의 시작 좌표(x, y)와 너비, 높이를 변수에 풀어넣음.
    undistorted_img = undistorted_img[y:y+h_roi, x:x+w_roi] # 왜곡을 펴면서 발생한 이미지 바깥쪽의 쓸모없는 검은색 둥근 여백들을 잘라내고, 꽉 찬 유효 이미지 부분만 배열 슬라이싱으로 남김.

    cv2.imshow("Original", test_img) # 렌즈 굴곡 때문에 선이 휘어져 있는 보정 전 원본 이미지를 'Original' 창에 띄움.
    cv2.imshow("Undistorted", undistorted_img) # 계산식을 통해 렌즈 왜곡이 바르게 펴지고 여백이 잘려나간 최종 이미지를 'Undistorted' 창에 띄워 비교하게 함.
    cv2.waitKey(0) # 사용자가 화면의 결과를 모두 확인하고 키보드의 아무 키나 누를 때까지 창을 끄지 않고 무한정 대기 상태를 유지함.
    cv2.destroyAllWindows() # 사용자가 키를 눌러 대기가 풀리면 열려있는 모든 이미지 창을 닫고 프로그램을 안전하게 종료함.