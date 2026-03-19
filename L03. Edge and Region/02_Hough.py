import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def solve_hough():
    # 1. 이미지 로드 및 복사
    src = cv.imread('dabo.jpg')
    if src is None:
        print("Error: 이미지를 불러올 수 없습니다.")
        return
    
    # 직선을 그릴 원본 이미지 복사본 생성
    display_img = src.copy()
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    # 2. Canny 에지 검출: cv.Canny()를 사용하여 에지 맵을 생성합니다
    # threshold1과 threshold2는 각각 100과 200으로 설정하여 노이즈를 줄이고 강한 에지만 남깁니다
    edges = cv.Canny(gray, 100, 200)

    # 3. 확률적 허프 변환: cv.HoughLinesP()를 사용하여 직선을 검출합니다
    # rho, theta, threshold, minLineLength, maxLineGap 값을 조정하여 직선 검출 성능을 개선합니다.
    # rho=1(1픽셀), theta=np.pi/180(1도) 해상도 사용. threshold=50으로 설정.
    # minLineLength=50 (선으로 인정할 최소 길이), maxLineGap=10 (끊어진 선을 연결할 최대 간격)
    lines = cv.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, 
                           minLineLength=50, maxLineGap=10)

    # 4. 검출된 직선 그리기: cv.line()을 사용하여 검출된 직선을 원본 이미지에 그립니다.
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 검출된 직선을 원본 이미지에서 빨간색으로 표시합니다.
            # cv.line()에서 색상은 (0, 0, 255) (빨간색)과 두께는 2로 설정합니다
            cv.line(display_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # 5. 시각화: Matplotlib를 사용하여 원본 이미지와 직선이 그려진 이미지를 나란히 시각화합니다.
    plt.figure(figsize=(12, 6))
    
    # 왼쪽: Canny 에지 검출 결과 (흑백 맵)
    plt.subplot(121)
    plt.imshow(edges, cmap='gray')
    plt.title('Canny Edge Map')
    
    # 오른쪽: 빨간색 선이 추가된 결과 이미지
    plt.subplot(122)
    plt.imshow(cv.cvtColor(display_img, cv.COLOR_BGR2RGB))
    plt.title('Detected Lines')
    
    plt.show()

if __name__ == "__main__":
    solve_hough()