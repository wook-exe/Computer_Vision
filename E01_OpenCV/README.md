# OpenCV 실습 (E01)

01. 이미지 불러오기 및 그레이스케일 변환
코드 : 01.py
핵심 코드 :
Python

# 이미지 로드
img = cv.imread('soccer.jpg') 

# BGR 이미지를 그레이스케일(흑백)로 변환
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 원본과 흑백 이미지를 가로로 연결하기 위해 흑백 이미지의 채널 맞추기
gray_3c = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

# 두 이미지를 가로로 나란히 연결하여 출력
result = np.hstack((img, gray_3c))


실행 결과:
![image](https://github.com/user-attachments/assets/7b4b8b38-95ce-4360-a54a-118615dddd45)


02. 페인팅 붓 크기 조절 기능 추가
마우스를 이용해 이미지 위에 그림을 그리고, 키보드를 이용해 붓의 크기를 실시간으로 조절
마우스 좌클릭은 파란색, 우클릭은 빨간색으로 연속해서 그릴 수 있습니다.
코드 : 02.py
핵심 코드 :
Python


# 마우스 이벤트 처리 콜백
def painting(event, x, y, flags, param):
    global BrushSiz

    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(img, (x,y), BrushSiz, LColor, -1)

    elif event == cv.EVENT_RBUTTONDOWN:
        cv.circle(img, (x,y), BrushSiz, RColor, -1)

    elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:
        cv.circle(img, (x,y), BrushSiz, LColor, -1)

    elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_RBUTTON:
        cv.circle(img, (x,y), BrushSiz, RColor, -1)

    cv.imshow('Painting', img)

# 창에 마우스 콜백 함수 연결
cv.setMouseCallback('Painting', painting)

# 키보드 입력 처리 루프
while True:
    key = cv.waitKey(1) & 0xFF

    if key == ord('+'):
        BrushSiz = min(MAX_BRUSH, BrushSiz + 1)
        print("Brush Size:", BrushSiz)

    elif key == ord('-'):
        BrushSiz = max(MIN_BRUSH, BrushSiz - 1)
        print("Brush Size:", BrushSiz)

    elif key == ord('q'):
        break


실행 결과:
![image](https://github.com/user-attachments/assets/2f5a56e6-9922-42b6-8deb-0ca2095012e6)

03. 마우스로 영역 선택 및 ROI(관심영역) 추출
이미지에서 원하는 부분을 마우스 드래그로 선택(ROI)하고, 해당 영역만 잘라내어 별도의 창에 띄우거나 저장
코드 : 03.py
핵심 코드 :
Python


# 마우스 드래그 중 사각형 그려서 영역 시각화
if event == cv.EVENT_MOUSEMOVE:
    if drawing:
        img = original_img.copy()
        cv.rectangle(img, start_pt, (x, y), (0, 255, 0), 2)

# 선택한 영역(ROI) 추출 (Numpy 슬라이싱 활용)
if x2 > x1 and y2 > y1:
    roi = original_img[y1:y2, x1:x2]
    cv.imshow('ROI', roi)

# 키보드 's' 입력 시 잘라낸 영역을 이미지 파일로 저장
if key == ord('s'):
    cv.imwrite('roi_saved.jpg', roi)


실행 결과:
![image](https://github.com/user-attachments/assets/f5d177ef-8936-4dbd-a13b-cd6045f1ac7c)
