import cv2 as cv 
import sys

img = cv.imread('soccer.jpg')

if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

# 붓 크기 설정
BrushSiz = 5                  
MIN_BRUSH = 1
MAX_BRUSH = 15

# 좌클릭(파란색), 우클릭(빨간색) 색상 설정
LColor, RColor = (255,0,0), (0,0,255)  

def painting(event, x, y, flags, param):
    global BrushSiz

    # 마우스 좌클릭 (파란색 점 찍기)
    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(img, (x,y), BrushSiz, LColor, -1)

    # 마우스 우클릭 (빨간색 점 찍기)
    elif event == cv.EVENT_RBUTTONDOWN:
        cv.circle(img, (x,y), BrushSiz, RColor, -1)

    # 마우스 좌클릭 드래그 (파란선)
    elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:
        cv.circle(img, (x,y), BrushSiz, LColor, -1)

    # 마우스 우클릭 드래그 (빨간선)
    elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_RBUTTON:
        cv.circle(img, (x,y), BrushSiz, RColor, -1)

    cv.imshow('Painting', img)


cv.namedWindow('Painting')  # 'Painting'이라는 이름의 윈도우 창 생성
cv.imshow('Painting', img)
cv.setMouseCallback('Painting', painting)


while True: # 무한 루프를 돌며 키보드 입력 처리
    key = cv.waitKey(1) & 0xFF
    
    if key == ord('+'): # '+' = 붓 크기를 1 증가
        BrushSiz = min(MAX_BRUSH, BrushSiz + 1)
        print("Brush Size:", BrushSiz)

    elif key == ord('-'):   # '-' = 붓 크기 1 감소
        BrushSiz = max(MIN_BRUSH, BrushSiz - 1)
        print("Brush Size:", BrushSiz)

    elif key == ord('q'):   # 무한 루프 종료
        break

cv.destroyAllWindows()