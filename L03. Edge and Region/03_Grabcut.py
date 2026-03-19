import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def solve_grabcut():
    # 1. 이미지 로드
    src = cv.imread('coffee_cup.jpg')
    if src is None:
        print("Error: 이미지를 불러올 수 없습니다.")
        return

    # 2. 초기 설정
    # 결과가 저장될 마스크 (원본 이미지의 높이, 너비와 동일한 크기의 0으로 채워진 배열)
    mask = np.zeros(src.shape[:2], np.uint8)
    
    # cv.grabCut()에서 bgdModel과 fgdModel은 np.zeros((1, 65), np.float64)로 초기화합니다.
    # 알고리즘 내부에서 배경과 전경의 히스토그램 모델을 저장하는 배열입니다.
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # 3. ROI(관심 영역) 설정: 초기 사각형 영역은 (x, y, width, height) 형식으로 설정합니다.
    # 이미지 테두리에서 약간 안쪽으로 들어간 사각형 영역을 객체가 있는 곳으로 추정합니다.
    h, w = src.shape[:2]
    rect = (20, 20, w-40, h-40) 

    # 4. GrabCut 실행: cv.grabCut()를 사용하여 대화식 분할을 수행합니다.
    # 사각형(rect) 정보를 바탕으로 5번 반복 연산(iterCount=5)하여 배경과 전경을 나눕니다.
    # cv.GC_INIT_WITH_RECT 옵션을 사용해 사각형 기반으로 초기화를 진행합니다.
    cv.grabCut(src, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)

    # 5. 마스크 처리: np.where()를 사용하여 마스크 값을 0 또는 1로 변경한 후 원본 이미지에 곱하여 배경을 제거합니다
    # 마스크 값은 cv.GC_BGD(0), cv.GC_FGD(1), cv.GC_PR_BGD(2), cv.GC_PR_FGD(3)를 사용합니다.
    # 값이 0(확실한 배경) 또는 2(아마도 배경)인 경우 0으로, 그 외(전경)는 1로 설정하여 마스크를 만듭니다.
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    
    # 6. 배경 제거: 마스크를 사용하여 원본 이미지에서 배경을 제거합니다
    # 컬러 이미지(3채널)와 곱하기 위해 mask2에 새로운 축을 추가하여 형태를 맞춥니다.
    result_img = src * mask2[:, :, np.newaxis]

    # 7. 시각화: matplotlib를 사용하여 원본 이미지, 마스크 이미지, 배경 제거 이미지 세 개를 나란히 시각화합니다
    titles = ['Original', 'GrabCut Mask', 'Extracted Object']
    
    # opencv는 BGR 형식을 사용하므로 matplotlib 출력을 위해 RGB로 변환합니다.
    images = [cv.cvtColor(src, cv.COLOR_BGR2RGB), mask2, cv.cvtColor(result_img, cv.COLOR_BGR2RGB)]

    plt.figure(figsize=(15, 5))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        # 마스크 이미지는 흑백(gray)으로 출력하고 나머지는 컬러로 출력합니다.
        plt.imshow(images[i], 'gray' if i==1 else None)
        plt.title(titles[i])
        plt.axis('off')
    plt.show()

if __name__ == "__main__":
    solve_grabcut()