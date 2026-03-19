import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def solve_sobel():
    # 1. 이미지 읽기: cv.imread()를 사용하여 이미지를 불러옵니다.
    src = cv.imread('edgeDetectionImage.jpg')
    if src is None:
        print("Error: 이미지를 찾을 수 없습니다. 경로를 확인해주세요.")
        return
    
    # 2. 그레이스케일 변환: cv.cvtColor()를 사용하여 그레이스케일로 변환합니다
    # 에지 검출은 밝기 변화가 중요하므로 색상 정보(BGR)를 제거합니다.
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    # 3. Sobel 필터 적용: cv.Sobel()을 사용하여 x축과 y축 방향의 에지를 각각 검출합니다
    # cv.CV_64F: 미분 시 음수 값이 잘리는 것을 막기 위해 64비트 부동소수점 자료형을 사용합니다.
    # x축 방향 에지 검출 (dx=1, dy=0)
    # ksize: 소벨 커널의 크기로 3 또는 5로 설정합니다
    sobel_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    
    # y축 방향 에지 검출 (dx=0, dy=1)
    sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)

    # 4. 에지 강도(Magnitude) 계산: cv.magnude()를 사용하여 에지 강도를 계산합니다
    # x축과 y축의 미분 결과를 벡터의 크기로 합산하여 전체 윤곽선을 구합니다.
    magnitude = cv.magnitude(sobel_x, sobel_y)
    
    # 5. 데이터 타입 변환: cv.convertScaleAbs()를 사용하여 에지 강도 이미지를 uint8로 변환합니다
    # 화면에 출력하기 위해 절댓값을 취하고 0~255 사이의 8비트 정수형으로 바꿉니다.
    sobel_combined = cv.convertScaleAbs(magnitude)

    # 6. 결과 시각화: Matplotlib를 사용하여 원본 이미지와 에지 강도 이미지를 나란히 시각화합니다.
    plt.figure(figsize=(12, 6))
    
    # 첫 번째 영역: 원본 이미지 (OpenCV의 BGR을 Matplotlib용 RGB로 변환하여 출력)
    plt.subplot(121)
    plt.imshow(cv.cvtColor(src, cv.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    # 두 번째 영역: 에지 결과 출력
    # plt.imshow()에서 cmap='gray'를 사용하여 흑백으로 시각화합니다
    plt.subplot(122)
    plt.imshow(sobel_combined, cmap='gray')
    plt.title('Sobel Edge Strength')
    plt.axis('off')

    # 레이아웃을 깔끔하게 정리하고 화면에 보여줍니다.
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    solve_sobel()