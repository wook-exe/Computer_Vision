import cv2 as cv
import numpy as np

drawing = False
start_pt = (-1, -1)
end_pt = (-1, -1)

# 이미지를 불러오고 화면에 출력
original_img = cv.imread('soccer.jpg')
img = original_img.copy()

def select_roi(event, x, y, flags, param):
    global drawing, start_pt, end_pt, img
    
    # 사용자가 클릭한 시작점에서 드래그하여 사각형을 그리며 영역을 선택
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        start_pt = (x, y)
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:
            img = original_img.copy()
            cv.rectangle(img, start_pt, (x, y), (0, 255, 0), 2) # 드래그 중인 영역을 시각화
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        end_pt = (x, y)
        cv.rectangle(img, start_pt, end_pt, (0, 255, 0), 2)
        
        # 마우스를 놓으면 해당 영역을 잘라내서 별도의 창에 출력
        x1, x2 = min(start_pt[0], end_pt[0]), max(start_pt[0], end_pt[0])
        y1, y2 = min(start_pt[1], end_pt[1]), max(start_pt[1], end_pt[1])
        
        if x2 > x1 and y2 > y1:
            roi = original_img[y1:y2, x1:x2]
            cv.imshow('ROI', roi)

cv.namedWindow('Image')
cv.setMouseCallback('Image', select_roi)

while True:
    cv.imshow('Image', img)
    key = cv.waitKey(1) & 0xFF
    
    if key == ord('r'): # 'r' = 영역 선택을 리셋하고 처음부터 다시 선택
        img = original_img.copy()
        try:
            cv.destroyWindow('ROI')
        except:
            pass
    elif key == ord('s'):   # 's' = 선택한 영역을 이미지 파일로 저장
        x1, x2 = min(start_pt[0], end_pt[0]), max(start_pt[0], end_pt[0])
        y1, y2 = min(start_pt[1], end_pt[1]), max(start_pt[1], end_pt[1])
        if x2 > x1 and y2 > y1:
            roi = original_img[y1:y2, x1:x2]
            
            cv.imwrite('roi_saved.jpg', roi)    # 이미지 저장
    elif key == ord('q'):
        break

cv.destroyAllWindows()