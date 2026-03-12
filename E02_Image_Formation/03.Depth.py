import cv2 # 영상 및 이미지 처리를 위한 다양한 내장 함수를 제공하는 OpenCV 라이브러리를 불러옴.
import numpy as np # 행렬 단위의 고속 수학 연산과 배열 슬라이싱 조작을 위해 NumPy 라이브러리를 불러옴.
from pathlib import Path # 파일 및 디렉토리 경로 문자열을 운영체제에 상관없이 안전한 객체로 다루기 위해 pathlib 모듈의 Path를 불러옴.

# 출력 폴더 생성
output_dir = Path("./outputs") # 현재 파이썬 코드가 실행되는 디렉토리(.) 아래에 'outputs'라는 이름의 하위 폴더 경로 객체를 하나 정의함.
output_dir.mkdir(parents=True, exist_ok=True) # 위에서 정의한 'outputs' 경로에 실제로 새 폴더를 생성함. 중간 경로가 없으면 만들고(parents=True), 이미 폴더가 있어도 에러를 무시함(exist_ok=True).

# 좌/우 이미지 불러오기
left_color = cv2.imread("c:/AIOSS/Computer_Vision/E02_Image_Formation/images/left.png") # OpenCV를 이용해 왼쪽 카메라 위치에서 찍은 컬러 시점 이미지를 절대 경로에서 읽어옴.
right_color = cv2.imread("c:/AIOSS/Computer_Vision/E02_Image_Formation/images/right.png") # 사람의 두 눈처럼 오른쪽 카메라 위치에서 약간 어긋나게 찍힌 컬러 이미지를 동일하게 읽어옴.

if left_color is None or right_color is None: # 만약 경로 오타나 파일 손상 등의 이유로 왼쪽 이미지나 오른쪽 이미지가 하나라도 메모리에 로드되지 않았다면
    raise FileNotFoundError("좌/우 이미지를 찾지 못했습니다.") # 사용자에게 어떤 파일이 문제인지 명확한 에러 로그를 띄우고 더 이상의 무의미한 연산을 강제 중단함.


# 카메라 파라미터
f = 700.0 # 캘리브레이션 과정을 통해 미리 구한 해당 카메라 렌즈의 초점 거리(Focal Length) 값을 픽셀(px) 단위 상수로 설정함.
B = 0.12 # 왼쪽 카메라와 오른쪽 카메라 렌즈 중심점 사이의 실제 물리적 떨어진 간격인 베이스라인(Baseline)을 미터(m) 단위(여기서는 0.12m = 12cm)로 설정함.

# ROI 설정
rois = { # 이미지 내에서 깊이(거리) 평균을 특별히 구하고 싶은 특정 사물들의 관심 영역(ROI) 픽셀 좌표들을 딕셔너리로 묶어 정의함.
    "Painting": (55, 50, 130, 110), # "Painting(벽에 걸린 그림)" 영역 사각형의 왼쪽 위 시작점(X:55, Y:50)과 그로부터의 너비(130), 높이(110) 길이를 설정함.
    "Frog": (90, 265, 230, 95), # "Frog(바닥의 개구리 인형)" 영역 사각형의 시작점(X:90, Y:265) 좌표와 사각형의 너비(230), 높이(95)를 설정함.
    "Teddy": (310, 35, 115, 90) # "Teddy(위쪽 선반의 곰인형)" 영역 사각형의 시작점(X:310, Y:35) 좌표와 너비(115), 높이(90)를 설정함.
}

# 그레이스케일 변환
left_gray = cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY) # 패턴 매칭 속도를 올리고 불필요한 색상 간섭을 없애기 위해 왼쪽 컬러 이미지를 흑백 밝기 이미지(Grayscale)로 변환함.
right_gray = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY) # 마찬가지 이유로 오른쪽 카메라의 컬러 이미지도 흑백 밝기 값만 가지는 이미지 배열로 변환함.

# -----------------------------
# 1. Disparity 계산
# -----------------------------
window_size = 5 # 좌/우 이미지 픽셀 매칭 시 주변 문맥(Context)을 파악하기 위해 비교할 픽셀 블록(윈도우)의 가로/세로 길이를 5로 설정함. (중심이 있어야 하므로 홀수 사용)
min_disp = 0 # 탐색할 최소 시차(Disparity, 어긋난 픽셀 수) 값을 0으로 설정함. 무한히 멀리 있는 배경 물체는 시차가 0에 가깝기 때문임.
num_disp = 16 * 5 # 가까운 물체를 탐색하기 위한 최대 시차 검색 범위를 설정함. SGBM 알고리즘 메모리 구조상 이 값은 반드시 16의 배수여야 하므로 80으로 세팅함.

stereo = cv2.StereoSGBM_create( # 글로벌 최적화를 흉내내어 정확도와 속도 밸런스가 좋은 SGBM(Semi-Global Block Matching) 스테레오 매칭 알고리즘 객체를 생성함.
    minDisparity=min_disp, # 생성하는 객체 내부의 탐색 시작 최소 시차 값을 우리가 위에서 정한 0으로 입력함.
    numDisparities=num_disp, # 생성하는 객체가 최소 시차로부터 탐색을 수행할 최대 시차 범위 개수(80)를 입력함.
    blockSize=window_size, # 블록 매칭 알고리즘이 픽셀 뭉치를 비교할 때 사용할 사각형 블록의 크기(5x5)를 객체에 전달함.
    P1=8 * 3 * window_size**2, # 이웃한 픽셀끼리 시차 값이 1픽셀 변할 때 부과할 페널티 가중치(평탄한 표면의 매끄러움 유지용) 공식을 계산하여 설정함.
    P2=32 * 3 * window_size**2, # 이웃 픽셀 시차가 2픽셀 이상 크게 확 뛸 때(보통 사물의 경계선) 부과할 페널티 가중치를 설정함. 이 값은 반드시 P1보다 커야 함.
    disp12MaxDiff=1, # 왼쪽을 기준으로 시차를 구한 결과와 오른쪽을 기준으로 구한 결과를 상호 비교할 때, 최대 1픽셀까지의 오차만 올바른 매칭으로 인정함.
    uniquenessRatio=10, # 최적 매칭 비용이 2순위 후보 매칭 비용보다 최소 10% 이상 월등히 좋아야만 최종 시차 값으로 확정하여 모호한 픽셀 노이즈를 방지함.
    speckleWindowSize=100, # 잘못 매칭되어 혼자 튀는 반점(Speckle, 노이즈)을 필터링하기 위해, 연결된 동일 시차 픽셀 덩어리의 최대 크기 임계값을 100픽셀로 설정함.
    speckleRange=32 # 픽셀 덩어리들이 하나로 이어져 있다고 인정할 최대 시차 변화 허용 범위를 설정하여 매끄러운 덩어리를 만듦.
)
# disparity 계산 (OpenCV는 결과를 16배 곱해서 반환하므로 16.0으로 나눔)
disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0 # 생성된 SGBM 객체에 좌/우 흑백 이미지를 넣고 연산하여 픽셀 시차 맵을 추출한 뒤, OpenCV 내부 표현 정밀도 때문에 강제로 곱해진 16을 다시 나누어 진짜 픽셀 시차(float32 실수형) 배열로 원상 복구함.

# -----------------------------
# 2. Depth 계산
# Z = fB / d
# -----------------------------
depth_map = np.zeros_like(disparity) # 계산된 실제 거리 값을 저장하기 위해 시차 맵(disparity)과 똑같은 해상도(모양)를 가지되, 모든 픽셀 값이 0으로 비워져 있는 새 배열을 생성함.
valid_mask = disparity > 0  # 시차 값이 0 이하인 부분은 매칭에 실패했거나 시야 밖 무한대 거리이므로, 이런 에러 값을 거르기 위해 시차가 0보다 큰(유효한) 픽셀 위치만 True로 반환하는 불리언 마스크 배열을 만듦.

# 유효한 disparity 픽셀에 대해서만 depth 계산
depth_map[valid_mask] = (f * B) / disparity[valid_mask] # 깊이 맵 배열의 유효한 픽셀 위치(마스크가 True인 곳)에만, 깊이 추정 공식(Z = 초점거리 * 카메라간격 / 시차)을 적용하여 미터(m) 단위의 실제 거리를 계산해 대입함.

# -----------------------------
# 3. ROI별 평균 disparity / depth 계산
# -----------------------------
results = {} # 관심 영역(ROI) 이름표와 그 영역 내부의 평균 시차, 평균 거리 계산 결과를 묶어 보관할 빈 딕셔너리 구조를 생성함.

for name, (x, y, w, h) in rois.items(): # 코드 윗부분에서 미리 좌표를 정의해 두었던 rois 딕셔너리에서 사물 이름(name)과 사각형 위치 정보(x, y, w, h)를 튜플로 한 묶음씩 반복해서 꺼내옴.
    # ROI 영역 추출
    roi_disp = disparity[y:y+h, x:x+w] # 전체 이미지 크기의 시차 맵 배열에서 현재 반복 중인 특정 사물의 사각형 영역 인덱스만큼만 NumPy 슬라이싱으로 잘라내어 추출함.
    roi_depth = depth_map[y:y+h, x:x+w] # 미터 단위가 기록된 전체 깊이 맵 배열에서도 위와 완벽히 동일한 좌표의 ROI 사각형 영역 부분만 잘라내어 추출함.

    # 해당 ROI 내에서 유효한 픽셀 마스크
    roi_valid_mask = roi_disp > 0 # 전체가 아닌 이 작게 잘라낸 영역(roi_disp) 안에서도 혹시 매칭에 실패해 시차가 0 이하인 노이즈 픽셀이 존재할 수 있으므로, 다시 0보다 큰 정상 픽셀만 걸러내는 마스크를 적용함.

    if np.any(roi_valid_mask): # 만약 우리가 잘라낸 ROI 픽셀들 중에 마스크를 통과한 정상 픽셀(True)이 단 하나라도 섞여 존재한다면
        mean_disp = np.mean(roi_disp[roi_valid_mask]) # ROI 안의 픽셀 중 정상적으로 계산된 픽셀들의 시차 값들만 모두 골라내어 더한 뒤 개수로 나누어 '평균 시차' 값을 도출함.
        mean_depth = np.mean(roi_depth[roi_valid_mask]) # 동일하게 ROI 안의 정상 픽셀들 위치의 깊이 값들만 골라내어 모두 더한 후 나누어 '평균 거리(m)' 값을 도출함.
    else: # 만약 ROI 박스를 잘못 쳐서 영역 안에 매칭에 성공한 정상 픽셀이 단 하나도 없다면 (오류 방지)
        mean_disp = 0.0 # 평균 시차를 계산할 수 있는 데이터가 없으므로 변수 에러를 막기 위해 임시로 0.0을 채워 넣음.
        mean_depth = 0.0 # 평균 거리 역시 계산할 수 없으므로 안전하게 임시 값 0.0을 채워 넣음.

    results[name] = {"mean_disp": mean_disp, "mean_depth": mean_depth} # 각 사물의 이름(예:"Teddy")을 딕셔너리 열쇠(Key)로 사용하고, 방금 안전하게 구한 평균 시차와 깊이 값을 그 안에 보관함.

# -----------------------------
# 4. 결과 출력
# -----------------------------
print(f"{'ROI Name':<10} | {'Mean Disparity (px)':<20} | {'Mean Depth (m)':<15}") # 콘솔 터미널 화면에 출력할 표의 상단 컬럼명들을 f-string 포맷팅을 사용해 폭을 맞춰 왼쪽 정렬(<)하여 깔끔하게 출력함.
print("-" * 55) # 컬럼 제목 부분과 내용 데이터 부분을 시각적으로 명확하게 구분하기 위해 하이픈(-) 문자를 55번 곱하여 가로로 긴 선을 그어줌.
for name, res in results.items(): # 방금 전 반복문을 통해 평균값 계산 결과가 누적 저장된 results 딕셔너리에서 물체 이름(name)과 결과 데이터 묶음(res)을 한 줄씩 꺼냄.
    print(f"{name:<10} | {res['mean_disp']:<20.4f} | {res['mean_depth']:<15.4f}") # 물체 이름은 10칸 여백, 평균 시차와 거리는 20/15칸 여백에 소수점 아래 4자리(.4f)까지만 반올림하여 표 양식에 맞게 정렬 출력함.

# -----------------------------
# 5. disparity 시각화
# 가까울수록 빨강 / 멀수록 파랑
# -----------------------------
disp_tmp = disparity.copy() # 원본 시차 배열 데이터를 훼손하지 않고 화면에 색상을 입히는 시각화 용도로만 사용하기 위해 배열의 메모리를 완전히 독립된 복사본(copy)으로 할당함.
disp_tmp[disp_tmp <= 0] = np.nan # 복사된 배열에서 매칭에 실패한 0 이하의 오류 픽셀들을, 이후 배열 최대/최소 백분위수 통계 계산 시 완전히 투명 취급하기 위해 NumPy 결측치 기호인 'NaN'으로 일괄 변환함.

if np.all(np.isnan(disp_tmp)): # 만약 시차 배열의 모든 픽셀 값이 매칭 실패 노이즈라서 배열 전체가 NaN 결측치로 덮여버린 최악의 상황이라면
    raise ValueError("유효한 disparity 값이 없습니다.") # 화면에 그릴 의미 있는 픽셀 정보가 1도 없으므로 치명적 값 오류(ValueError)를 발생시키고 코드를 즉시 종료함.

d_min = np.nanpercentile(disp_tmp, 5) # 색상 매핑 시 극도로 심한 노이즈 값이 전체 스케일을 망치는 것을 막기 위해, 시차 값들을 줄 세웠을 때 하위 5% 위치의 값을 최소 시차 기준치(d_min)로 설정함. (NaN 무시)
d_max = np.nanpercentile(disp_tmp, 95) # 마찬가지로 극도로 높은 오류 픽셀을 거르기 위해, 상위 5% 위치(전체의 95% 백분위수) 값을 최대 시차 기준치(d_max)로 설정하여 아웃라이어를 잘라냄.

if d_max <= d_min: # 만약 벽면같이 밋밋한 곳만 찍혀서 모든 픽셀 시차가 거의 같아 하위 5% 최소치와 상위 5% 최대치가 같아지는 현상이 발생했다면
    d_max = d_min + 1e-6 # 바로 다음 정규화 단계에서 분모가 0이 되어 나누기 에러(ZeroDivisionError)가 터지는 것을 막기 위해 최대치에 아주 미세한 소수(0.000001)를 억지로 더해줌.

disp_scaled = (disp_tmp - d_min) / (d_max - d_min) # 모든 픽셀의 시차 값에서 최소 기준치를 빼고 전체 범위(최대-최소)로 나누어, 뒤죽박죽이던 실수 시차 값들을 0.0부터 1.0 사이의 비율 값으로 스케일링(정규화)함.
disp_scaled = np.clip(disp_scaled, 0, 1) # 정규화 수식 과정에서 아까 거르지 못한 5% 밖의 극단적 노이즈 값들이 0보다 작아지거나 1보다 커졌을 테니, 이들을 강제로 0과 1 테두리로 자름(clip).

disp_vis = np.zeros_like(disparity, dtype=np.uint8) # 0.0~1.0 비율을 시각적 색상으로 바꾸기 전, 일단 0~255 사이의 흑백 음영 픽셀 값을 담을 수 있는 8비트 정수형(uint8) 빈 검은색 캔버스를 준비함.
valid_disp = ~np.isnan(disp_tmp) # 에러 처리용으로 만들었던 NaN 결측치 픽셀을 제외한(~) 즉, 정상적으로 시차 숫자가 들어있는 유효 픽셀 좌표 위치들만 True로 켜지는 시각화용 마스크를 생성함.
disp_vis[valid_disp] = (disp_scaled[valid_disp] * 255).astype(np.uint8) # 0.0~1.0 사이를 오가던 정상 픽셀 비율 값에 255를 곱해 0~255 사이 흑백 명도 값으로 부풀린 뒤, 소수점을 버린 정수로 변환하여 캔버스를 칠함.

disparity_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET) # 완성된 흑백 명도 캔버스 이미지에 OpenCV의 'JET' 컬러맵 필터를 입혀, 밝은 픽셀(시차가 큼 = 가까움)은 빨간색으로, 어두운 픽셀은 파란색 히트맵으로 매핑함.

# -----------------------------
# 6. depth 시각화
# 가까울수록 빨강 / 멀수록 파랑
# -----------------------------
depth_vis = np.zeros_like(depth_map, dtype=np.uint8) # 깊이(Depth) 값도 화면에 컬러 히트맵으로 그리기 위해 시차 때와 똑같이 0~255 정수 밝기 값을 담을 빈 검은색 캔버스를 하나 더 생성함.

if np.any(valid_mask): # 만약 거리 계산 공식(fB/d)을 정상적으로 통과한 유효한 깊이 픽셀 마스크(True)가 이미지 안에 하나라도 제대로 살아있다면
    depth_valid = depth_map[valid_mask] # 미터(m) 단위가 담긴 원본 깊이 배열에서 에러난 픽셀들은 버리고 유효한 위치의 진짜 깊이 값들만 쭉 뽑아내어 1차원 배열로 평탄화하여 모아둠.

    z_min = np.percentile(depth_valid, 5) # 시차 시각화 때와 동일한 논리로 노이즈의 악영향을 막기 위해 추출된 진짜 깊이 값들 중 하위 5% 위치의 가장 가까운 거리 값을 최소치(z_min)로 기준 잡음.
    z_max = np.percentile(depth_valid, 95) # 추출된 진짜 깊이 값들 중 상위 5% 위치의 가장 먼 거리 값을 최대치(z_max)로 기준 잡아 아웃라이어 간섭을 차단함.

    if z_max <= z_min: # 만약 측정된 거리 값이 전부 동일해서 최대치와 최소치가 같아지는 예외 상황이 발생한다면
        z_max = z_min + 1e-6 # 역시 정규화 과정의 0 나누기(ZeroDivisionError) 충돌을 방지하고자 최대 깊이 값에 미세한 소수점을 강제로 추가하여 차이를 벌려줌.

    depth_scaled = (depth_map - z_min) / (z_max - z_min) # 각 픽셀에 기록된 찐 실제 거리 값을 최소치와 최대치 범위 기준으로 계산하여 0.0에서 1.0 사이의 정규화된 스케일 비율 값으로 압축함.
    depth_scaled = np.clip(depth_scaled, 0, 1) # 압축 과정에서 범위를 벗어나 0 밑으로 떨어지거나 1 위로 솟은 비정상 픽셀 비율 값들을 안전하게 0과 1 바운더리로 강제로 깎아냄.

    # depth는 클수록 멀기 때문에 반전
    depth_scaled = 1.0 - depth_scaled # *핵심 코드*: 시차와 반대로 실제 거리(Depth)는 숫자가 클수록 멀리 있다는 뜻임. 직관적으로 가까운 것을 빨갛게(큰 값) 칠하기 위해 1.0에서 빼주어 거리를 역전(반전)시킴.
    depth_vis[valid_mask] = (depth_scaled[valid_mask] * 255).astype(np.uint8) # 역전되어 0.0~1.0을 가지는 비율 값에 255를 곱해 0~255 픽셀 밝기 수치로 바꾼 뒤 소수점을 뗀 정수로 변환하여 캔버스의 유효 위치에만 채색함.

depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET) # 완성된 0~255의 흑백 깊이 명도 캔버스에 'JET' 컬러맵을 덧칠함. 값이 크면(원래 가까운 거리를 반전시켰으므로 가까운 사물) 빨갛게 색칠됨.

# -----------------------------
# 7. Left / Right 이미지에 ROI 표시
# -----------------------------
left_vis = left_color.copy() # 원본 왼쪽 컬러 이미지 배열을 파괴하지 않고 ROI 사각형 안내선을 그리기 위해 배열 복사본을 만들어 시각화 캔버스로 사용함.
right_vis = right_color.copy() # 원본 오른쪽 컬러 이미지 위에도 동일하게 ROI 영역 안내선을 표시하기 위해 배열의 독립적인 복사본을 생성함.

for name, (x, y, w, h) in rois.items(): # 코드 상단에 정의해둔 관심 영역(ROI) 딕셔너리에서 물건의 이름과 박스를 칠 좌표 변수들을 한 꾸러미씩 순회하며 꺼냄.
    cv2.rectangle(left_vis, (x, y), (x + w, y + h), (0, 255, 0), 2) # 왼쪽 카메라 이미지 복사본 캔버스에 (x,y) 시작점부터 우하단 대각선 끝점까지 선 두께 2인 초록색(B=0, G=255, R=0) 사각형 박스를 그림.
    cv2.putText(left_vis, name, (x, y - 8), # 방금 그린 왼쪽 이미지 초록 사각형 박스 약간 위쪽(y - 8) 허공 위치를 글자 작성 시작점으로 잡고 물건 이름 문자열을 새김.
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2) # 새길 글자의 폰트 종류(기본 고딕 형태), 글자 크기(0.6배), 글자 색상(초록색), 글자 선 두께(2) 파라미터를 지정함.

    cv2.rectangle(right_vis, (x, y), (x + w, y + h), (0, 255, 0), 2) # 오른쪽 카메라 이미지 복사본 캔버스에도 왼쪽과 완벽히 동일한 좌표와 크기로 초록색 사각형 안내선 박스를 그림.
    cv2.putText(right_vis, name, (x, y - 8), # 오른쪽 이미지에 그려진 초록색 박스 약간 위쪽(y - 8) 공간에 마찬가지로 해당 물체의 이름 텍스트를 입력함.
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2) # 글자 폰트, 크기, 색상, 두께 등을 앞서 작성한 왼쪽 이미지 글자 설정과 완벽하게 동일하게 맞춤.

# -----------------------------
# 8. 저장
# -----------------------------
cv2.imwrite(str(output_dir / "disparity_color.png"), disparity_color) # 가장 윗부분에서 안전하게 만들어둔 'outputs' 폴더 경로 아래에 화려한 시차 히트맵 이미지를 "disparity_color.png" 물리 파일로 압축해 디스크에 저장함.
cv2.imwrite(str(output_dir / "depth_color.png"), depth_color) # 'outputs' 폴더 경로 안에 반전된 실제 거리 히트맵 이미지를 "depth_color.png"라는 이름의 그림 파일 형식으로 디스크에 안전하게 구움.
cv2.imwrite(str(output_dir / "left_rois.png"), left_vis) # 물체의 이름과 초록색 ROI 사각형 안내선이 칠해진 왼쪽 카메라 시각화 이미지를 "left_rois.png" 파일명으로 로컬 폴더에 저장함.
cv2.imwrite(str(output_dir / "right_rois.png"), right_vis) # 물체 이름표와 초록 사각형이 입혀진 오른쪽 카메라 시각화 이미지를 "right_rois.png" 파일명으로 동일한 폴더에 일괄 저장함.

# -----------------------------
# 9. 출력
# -----------------------------
cv2.imshow("Disparity Heatmap", disparity_color) # OpenCV의 창 생성 GUI 함수를 이용해 디스크에 저장한 시차 히트맵 배열을 모니터 화면의 "Disparity Heatmap"이라는 제목 팝업 창에 띄워줌.
cv2.imshow("Depth Heatmap", depth_color) # 메모리에 계산된 거리 기반 컬러 히트맵 배열 이미지를 "Depth Heatmap"이라는 이름표가 달린 모니터 팝업 창에 나란히 띄워 사용자에게 보여줌.
cv2.imshow("Left ROI", left_vis) # 어느 영역의 평균 거리를 계산했는지 알 수 있도록 초록 박스가 쳐진 왼쪽 카메라 원본 이미지를 "Left ROI"라는 팝업 창에 같이 띄움.
cv2.imshow("Right ROI", right_vis) # 어느 영역의 평균 거리를 계산했는지 확인 가능하게 초록 박스가 쳐진 오른쪽 카메라 이미지를 "Right ROI" 팝업 창에 띄워 총 4개의 창을 보여줌.

cv2.waitKey(0) # 팝업 창들이 번쩍하고 켜졌다 바로 꺼지지 않도록, 사용자가 결과 화면들을 눈으로 모두 확인하고 키보드 아무 키(Space, Enter 등)나 누를 때까지 파이썬 실행 사이클을 영구 정지 상태로 홀딩함.
cv2.destroyAllWindows() # 사용자의 키 입력이 감지되면 대기 홀딩이 풀리고, 메모리 누수를 막기 위해 현재 모니터 화면에 켜져 있는 4개의 OpenCV GUI 팝업 창들을 한 번에 모두 강제 종료함.