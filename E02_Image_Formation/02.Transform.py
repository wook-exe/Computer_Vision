import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread("rose.png")
if img is None:
    raise FileNotFoundError("rose.png 이미지를 찾지 못했습니다.")

rows, cols = img.shape[:2]

# 조건
angle = 30          # +30도 회전
scale = 0.8         # 크기 0.8배
tx = 80             # x축 +40px
ty = -40            # y축 -20px

# 이미지 중심
center = (cols / 2, rows / 2)

# 회전 + 스케일 행렬 생성
M = cv.getRotationMatrix2D(center, angle, scale)

# 평행이동 추가
M[0, 2] += tx
M[1, 2] += ty

# 원본 크기 유지
transformed_img = cv.warpAffine(
    img,
    M,
    (cols, rows),
    flags=cv.INTER_LINEAR,
    borderMode=cv.BORDER_CONSTANT,
    borderValue=(0, 0, 0)
)

# 시각화
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv.cvtColor(transformed_img, cv.COLOR_BGR2RGB))
plt.title("Rotated + Scaled + Translated")
plt.axis("off")

plt.show()