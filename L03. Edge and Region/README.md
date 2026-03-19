# L03 Edge and Region 실습

## 📑 목차

1.  [Sobel 에지 검출 (01_Sobel_edge.py)]
2.  [허프 변환 직선 검출 (02_Hough.py)]
3.  [GrabCut 객체 분할 (03_Grabcut.py)]

## 1\. Sobel 에지 검출 (01\_Sobel\_edge.py)

과제 목표: 소벨 에지 검출 및 결과 시각화. `edgeDetectionImage` 이미지를 그레이스케일로 변환한 후, Sobel 필터를 사용해 x축과 y축 방향의 에지를 검출하고 에지 강도를 시각화합니다.

### 💡 핵심 로직

  이미지 로드 및 변환: `cv.imread()`를 사용하여 이미지를 불러온 뒤, `cv.cvtColor()`를 통해 그레이스케일로 변환합니다.
  Sobel 필터 적용: `cv.Sobel()`을 사용하여 x축(`cv.CV_64F`, 1, 0)과 y축(`cv.CV_64F`, 0, 1) 방향의 에지를 각각 추출하며, 커널 크기(`ksize`)는 3 또는 5로 설정합니다.
  에지 강도 계산 및 변환: `cv.magnitude()`를 사용하여 전체 에지 강도를 계산하고, `cv.convertScaleAbs()`를 이용해 화면에 표시할 수 있는 `uint8` 형식으로 변환합니다.
  결과 시각화: Matplotlib의 `plt.imshow()`에서 `cmap='gray'`를 설정하여 원본 이미지와 흑백 에지 강도 이미지를 나란히 출력합니다.

### 💻 전체 코드

```python
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
```
-----

### 🖼️ 결과
![image](https://github.com/user-attachments/assets/f6bd9345-6558-461e-8ddc-3571e347f39e)

## 2\. 캐니 에지 및 허프 변환 직선 검출 (02\_Hough.py)

과제 목표: 캐니 에지 및 허프 변환을 이용한 직선 검출. `dabo` 이미지에 캐니 에지 검출을 적용해 에지 맵을 생성하고, 허프 변환으로 추출한 직선을 원본 이미지에 빨간색으로 표시합니다

### 💡 핵심 로직

   Canny 에지 검출: `cv.Canny()`를 사용하여 에지 맵을 생성하며, 임계값(`threshold1`, `threshold2`)은 각각 100과 200으로 설정합니다
   직선 검출: `cv.HoughLinesP()` 알고리즘을 사용하며, 직선 검출 성능 개선을 위해 `rho`, `theta`, `threshold`, `minLineLength`, `maxLineGap` 파라미터를 조절합니다
   직선 그리기 및 시각화: 검출된 좌표를 바탕으로 `cv.line()`을 사용해 색상 (0, 0, 255)의 빨간색과 두께 2인 선을 원본에 그립니다. 이후 Matplotlib를 사용해 나란히 시각화합니다

### 💻 전체 코드

```python
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
```

-----

### 🖼️ 결과
![image](https://github.com/user-attachments/assets/f6e2d769-79f5-4e96-a850-cb83abe8b49f)

## 3\. GrabCut 객체 분할 (03\_Grabcut.py)

과제 목표: GrabCut을 이용한 대화식 영역 분할 및 객체 추출. `coffee cup` 이미지에서 지정한 사각형을 바탕으로 알고리즘을 수행하여 배경을 제거하고, 객체 추출 결과를 마스크 형태와 객체만 남은 이미지로 각각 출력합니다

### 💡 핵심 로직

   초기 설정 및 분할 수행: 분할을 위해 `cv.grabCut()` 알고리즘을 사용합니다. 초기 사각형 영역은 `(x, y, width, height)` 형식으로 지정하며, 내부에서 활용되는 `bgdModel`과 `fgdModel`은 `np.zeros((1, 65), np.float64)` 영행렬로 초기화합니다.
   배경 제거 처리: 마스크 값(`cv.GC_BGD`, `cv.GC_FGD` 등)을 바탕으로 확실한 배경과 전경을 구분합니다. `np.where()` 함수를 활용하여 마스크 값을 0(배경) 또는 1(전경)로 변경한 뒤 원본 이미지에 곱하여 배경을 검게 제거합니다.
   다중 시각화: Matplotlib를 사용하여 원본 이미지, 마스크 이미지, 최종 배경 제거 이미지를 세 개 나란히 시각화합니다.

### 💻 전체 코드

```python
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
    result_img = src  mask2[:, :, np.newaxis]

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
```

### 🖼️ 결과
![image](https://github.com/user-attachments/assets/9a74f918-abcc-4a84-80cf-4fabf47e2175)