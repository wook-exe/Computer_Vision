import cv2 as cv
import sys
import numpy as np

img = cv.imread('soccer.jpg')    # 이미지 로드

if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 이미지를 그레이스케일로 변환

gray_3c = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)  # 1채널을 3채널 형태로 변환

# 원본 이미지와 그레이스케일(흑백) 이미지를 가로로 연결하여 출력
imgs = np.hstack((img, gray_3c))

cv.imshow('Result', imgs)   # 화면에 결과 표시
cv.waitKey(0)   # 아무 키나 누르면 창이 닫힘
cv.destroyAllWindows()

'''
cv.namedWindow('Result', cv.WINDOW_NORMAL) 

cv.imshow('Result', imgs)
cv.waitKey(0)
cv.destroyAllWindows()  # 해상도 오류로 사진이 짤릴 경우 창 조절을 가능하게 만들어 해결
'''