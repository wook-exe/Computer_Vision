# L06. Dynamic Vision

## 📑 목차
1. [SORT 알고리즘을 활용한 다중 객체 추적기 (01_Sort.py)]
2. [Mediapipe를 활용한 얼굴 랜드마크(468 Points) 추출 및 시각화 (02_Mediapipe.py)]


## 🛠 환경 설정 및 실행 방법

본 프로젝트는 **Python 3.10** 환경에서 테스트되었습니다. 터미널에서 아래 명령어를 순서대로 입력하여 가상환경을 구축하고 실행하십시오.

```bash
# 1. 가상환경 생성 (venv 사용 시)
python -m venv vision_env
source vision_env/bin/activate  # Mac/Linux
vision_env\Scripts\activate     # Windows

# 2. 필수 패키지 설치
# 주의: 의존성 충돌 방지를 위해 아래 명령어로 설치를 권장함
pip install opencv-python numpy mediapipe filterpy
pip uninstall tensorflow -y  # Mediapipe 충돌 방지

# 3. 모델 파일 준비 (과제 01 실행 전)
# yolov3.weights와 yolov3.cfg 파일이 소스 코드와 같은 폴더에 있어야 함
```

-----

## 1\. SORT 알고리즘을 활용한 다중 객체 추적기 (01_Sort.py)


과제 목표

  * YOLOv3 객체 검출기와 SORT(Simple Online and Realtime Tracking) 알고리즘을 결합하여 비디오 내 이동 객체에 고유 ID를 부여하고 실시간으로 추적함.
  * 객체 추적의 핵심 개념인 데이터 연관(Data Association)과 상태 예측(State Prediction) 과정을 이해함.

### 💡 핵심 로직

1.  **YOLOv3 Detection:** OpenCV DNN 모듈을 사용하여 프레임별로 객체를 검출하고 NMS(Non-Maximum Suppression)를 통해 최적의 Bounding Box를 선별함.
2.  **Kalman Filter:** SORT 내부에서 칼만 필터를 사용하여 이전 프레임의 객체 위치를 기반으로 현재 프레임의 위치를 예측함.
3.  **Hungarian Algorithm:** 예측된 위치와 실제 검출된 위치 간의 IOU(Intersection over Union)를 계산하여 동일 객체 여부를 판단하고 고유 ID를 유지함.

### 💻 전체 코드

```python
import cv2 # OpenCV 라이브러리를 불러와서 이미지 및 비디오 처리를 수행함.
import numpy as np # 수치 연산 및 배열 처리를 위해 Numpy 라이브러리를 불러옴.
from sort import Sort # 객체 추적을 수행하기 위해 SORT 알고리즘 모듈을 불러옴.

def main(): # 프로그램의 메인 실행 함수를 정의함.
    # YOLOv3 모델 가중치와 설정 파일 경로를 지정함 (실제 파일 경로에 맞게 수정 필요).
    weights_path = "yolov3.weights" # YOLOv3 가중치 파일 경로를 문자열로 저장함.
    config_path = "yolov3.cfg" # YOLOv3 네트워크 설정 파일 경로를 문자열로 저장함.
    
    # OpenCV DNN 모듈을 사용하여 YOLO 모델을 메모리에 로드함.
    net = cv2.dnn.readNet(weights_path, config_path) # 지정된 경로의 파일로 딥러닝 네트워크 객체를 생성함.
    
    # 네트워크에서 사용할 출력 레이어의 이름들을 가져옴.
    layer_names = net.getLayerNames() # 네트워크의 모든 레이어 이름을 리스트로 가져옴.
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()] # 연결되지 않은 출력 레이어(최종 출력)의 이름만 추출함.
    
    # 분석할 입력 비디오 파일을 엶 (웹캠 사용 시 0으로 변경).
    cap = cv2.VideoCapture("slow_traffic_small.mp4") # VideoCapture 객체를 생성하여 비디오 스트림을 오픈함.
    
    # SORT 추적기 객체를 초기화함 (최대 1개 프레임 누락 허용, IOU 임계값 0.3).
    tracker = Sort(max_age=1, min_hits=3, iou_threshold=0.3) # SORT 인스턴스를 생성하여 추적을 준비함.
    
    while cap.isOpened(): # 비디오가 정상적으로 열려있는 동안 무한 루프를 실행함.
        ret, frame = cap.read() # 비디오에서 한 프레임을 읽어옴 (ret은 성공 여부, frame은 이미지 데이터).
        if not ret: # 프레임을 제대로 읽지 못했을 경우(비디오가 끝났거나 에러) 내부를 실행함.
            break # while 루프를 탈출해 비디오 처리를 종료함.
            
        height, width, channels = frame.shape # 현재 프레임 이미지의 높이, 너비, 채널 수를 추출함.
        
        # YOLO 모델에 입력하기 위해 이미지를 4차원 Blob 형태로 변환함.
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False) # 스케일링, 크기 조정, BGR->RGB 변환을 수행함.
        net.setInput(blob) # 변환된 Blob을 네트워크의 입력으로 설정함.
        outs = net.forward(output_layers) # 네트워크를 순방향으로 실행하여 객체 검출 결과를 얻어옴.
        
        class_ids = [] # 검출된 객체의 클래스 ID를 저장할 빈 리스트를 생성함.
        confidences = [] # 검출된 객체의 신뢰도(확률)를 저장할 빈 리스트를 생성함.
        boxes = [] # 검출된 객체의 Bounding Box 좌표를 저장할 빈 리스트를 생성함.
        
        for out in outs: # 각 출력 레이어의 결과값들을 순회함.
            for detection in out: # 하나의 출력 레이어 내에 있는 각 검출 결과(객체)를 순회함.
                scores = detection[5:] # 처음 5개 값(좌표, 객체 여부) 이후의 값들이 각 클래스별 확률 수치임.
                class_id = np.argmax(scores) # 확률 수치 중 가장 큰 값의 인덱스(클래스 ID)를 찾음.
                confidence = scores[class_id] # 가장 큰 확률 값 자체를 신뢰도로 저장함.
                
                if confidence > 0.5: # 신뢰도가 0.5 (50%)를 초과하는 유의미한 객체만 필터링함.
                    center_x = int(detection[0] * width) # 중심 X 좌표를 이미지 실제 너비에 곱해 픽셀 단위로 복원함.
                    center_y = int(detection[1] * height) # 중심 Y 좌표를 이미지 실제 높이에 곱해 픽셀 단위로 복원함.
                    w = int(detection[2] * width) # 박스 너비를 이미지 실제 너비에 곱해 픽셀 단위로 복원함.
                    h = int(detection[3] * height) # 박스 높이를 이미지 실제 높이에 곱해 픽셀 단위로 복원함.
                    
                    x = int(center_x - w / 2) # 중심 좌표와 너비를 이용해 좌상단 X 좌표를 계산함.
                    y = int(center_y - h / 2) # 중심 좌표와 높이를 이용해 좌상단 Y 좌표를 계산함.
                    
                    boxes.append([x, y, w, h]) # 계산된 좌표 정보를 boxes 리스트에 추가함.
                    confidences.append(float(confidence)) # 신뢰도 값을 float 형태로 변환하여 리스트에 추가함.
                    class_ids.append(class_id) # 해당 객체의 클래스 ID를 리스트에 추가함.
                    
        # Non-Maximum Suppression을 적용하여 겹치는 Bounding Box 중 확률이 가장 높은 것만 남김.
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4) # 신뢰도 임계값 0.5, IOU 임계값 0.4를 사용함.
        
        # SORT 추적기에 입력할 형태로 데이터를 가공하기 위한 빈 리스트를 생성함.
        dets = [] # [x1, y1, x2, y2, score] 형태의 데이터를 담을 리스트임.
        
        if len(indexes) > 0: # NMS를 통과한 객체가 하나라도 존재할 경우 내부를 실행함.
            for i in indexes.flatten(): # indexes 배열을 1차원으로 펴서 각 인덱스를 순회함.
                x, y, w, h = boxes[i] # 해당 인덱스의 Bounding Box 좌표와 크기를 가져옴.
                dets.append([x, y, x + w, y + h, confidences[i]]) # 좌상단 좌표, 우하단 좌표, 신뢰도를 dets에 추가함.
                
        dets = np.array(dets) # 생성된 Python 리스트를 Numpy 배열로 변환해 SORT 입력 규격을 맞춤.
        
        if len(dets) == 0: # 만약 이번 프레임에서 검출된 객체가 전혀 없을 경우 내부를 실행함.
            dets = np.empty((0, 5)) # SORT 알고리즘의 오류 방지를 위해 0행 5열의 빈 Numpy 배열을 생성함.
            
        # 가공된 검출 데이터를 SORT 추적기에 전달하여 객체들의 현재 상태(좌표 및 ID)를 업데이트 받음.
        trackers = tracker.update(dets) # trackers 변수에는 [x1, y1, x2, y2, id] 형태의 배열이 반환됨.
        
        for trk in trackers: # 추적기에서 반환된 각 추적 객체의 정보들을 순회함.
            x1, y1, x2, y2, obj_id = [int(v) for v in trk] # 좌표 및 ID 데이터를 정수형으로 변환해 언패킹함.
            
            # 추적된 객체의 위치를 화면에 파란색 Bounding Box로 그림 (두께 2).
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2) # 이미지 상에 사각형을 렌더링함.
            # 추적된 객체의 고유 ID 텍스트를 구성함.
            text = f"ID: {obj_id}" # 화면에 표시할 문자열 포맷을 지정함.
            # 객체 Box 위쪽에 고유 ID 텍스트를 빨간색으로 출력함.
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2) # 이미지 상에 텍스트를 렌더링함.
            
        # 처리가 완료된 현재 프레임 이미지를 화면에 띄워 보여줌.
        cv2.imshow("Multi-Object Tracking", frame) # 윈도우 창 이름을 지정하고 이미지를 표시함.
        
        # 1ms 대기하며 'Esc' 키(27)가 눌렸는지 확인함.
        if cv2.waitKey(1) & 0xFF == 27: # 눌린 키의 ASCII 코드가 27일 경우 내부를 실행함.
            break # while 루프를 탈출해 프로그램을 종료함.
            
    # 모든 작업이 끝나면 비디오 파일(또는 웹캠) 점유를 해제함.
    cap.release() # 자원을 시스템에 반환함.
    # 화면에 띄워진 모든 OpenCV 윈도우 창을 닫음.
    cv2.destroyAllWindows() # 메모리 낭비를 막기 위해 창을 종료함.

if __name__ == "__main__": # 이 스크립트가 직접 실행될 때만 아래 블록을 실행함.
    main() # 메인 함수를 호출해 프로그램을 시작함.
```

![image]()

-----

## 2\. Mediapipe를 활용한 얼굴 랜드마크(468 Points) 추출 및 시각화 (02_Mediapipe.py)

과제 목표

  * Mediapipe의 FaceMesh 솔루션을 사용하여 실시간 영상에서 얼굴의 고유 특징점 468개를 추출함.
  * 정규화된 좌표계를 픽셀 좌표계로 변환하는 방법과 실시간 시각화 기법을 습득함.

### 💡 핵심 로직

1.  **FaceMesh Initialization:** `refine_landmarks=False` 설정을 통해 PDF 요구사항인 기본 468개 랜드마크 모드로 초기화함.
2.  **Coordinate Transformation:** 모델에서 반환된 0.0\~1.0 사이의 정규화된 좌표에 이미지의 실제 가로/세로 길이를 곱해 픽셀 위치를 계산함.
3.  **Visualization:** 계산된 좌표에 `cv2.circle`을 활용하여 점을 찍고, 사용자 편의를 위해 `cv2.flip`으로 거울 모드를 적용함.

### 💻 전체 코드

```python
import cv2 # OpenCV 라이브러리를 임포트하여 실시간 웹캠 영상 처리 및 화면 출력을 수행함.
import mediapipe as mp # 구글의 Mediapipe 라이브러리를 임포트하여 AI 기반 얼굴 랜드마크 추출 기능을 사용함.

def main(): # 프로그램의 핵심 실행 로직을 담은 main 함수를 정의함.
    mp_face_mesh = mp.solutions.face_mesh # Mediapipe의 FaceMesh 모듈을 사용하기 쉽게 변수에 할당함.
    
    cap = cv2.VideoCapture(0) # 기본 웹캠(0번)을 호출하여 실시간 비디오 스트림 객체를 생성함.
    
    # PDF 요구사항에 명시된 468개 랜드마크 추출을 위해 설정을 최적화함.
    with mp_face_mesh.FaceMesh( 
        static_image_mode=False, # 실시간 추적을 위해 비디오 스트림 모드를 활성화함.
        max_num_faces=1, # 탐지 대상을 얼굴 1개로 제한하여 연산 속도를 최적화함.
        refine_landmarks=False, # PDF 요구사항에 맞춰 눈동자 제외 기본 468개 포인트만 추출함.
        min_detection_confidence=0.5, # 얼굴 검출을 위한 최소 신뢰도를 0.5로 설정함.
        min_tracking_confidence=0.5 # 랜드마크 추적 유지를 위한 최소 신뢰도를 0.5로 설정함.
    ) as face_mesh: 
    
        while cap.isOpened(): # 웹캠이 연결된 동안 루프를 실행함.
            success, image = cap.read() # 프레임을 읽어와 성공 여부와 이미지를 저장함.
            if not success: 
                continue # 프레임 읽기 실패 시 다음 시도로 넘어감.
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # 모델 입력을 위해 BGR을 RGB로 변환함.
            image_rgb.flags.writeable = False # 처리 속도 향상을 위해 메모리를 읽기 전용으로 설정함.
            results = face_mesh.process(image_rgb) # 얼굴 랜드마크 추출 연산을 수행함.
            
            image_rgb.flags.writeable = True # 시각화를 위해 다시 쓰기 권한을 부여함.
            ih, iw, _ = image.shape # 정규화 좌표 복원을 위해 이미지의 실제 픽셀 크기를 가져옴.
            
            if results.multi_face_landmarks: # 랜드마크가 검출되었을 경우 시각화를 시작함.
                for face_landmarks in results.multi_face_landmarks: 
                    for landmark in face_landmarks.landmark: 
                        x = int(landmark.x * iw) # 정규화된 x좌표를 실제 픽셀 좌표로 변환함.
                        y = int(landmark.y * ih) # 정규화된 y좌표를 실제 픽셀 좌표로 변환함.
                        cv2.circle(image, (x, y), 1, (0, 255, 0), -1) # 해당 위치에 초록색 점을 그림.
                        
            cv2.imshow('Mediapipe FaceMesh', cv2.flip(image, 1)) # 거울 모드로 반전시켜 화면에 표시함.
            
            if cv2.waitKey(5) & 0xFF == 27: # PDF 요구사항에 따라 ESC 키(27) 입력 시 루프를 종료함.
                break 
                
    cap.release() # 웹캠 자원을 반환함.
    cv2.destroyAllWindows() # 모든 윈도우 창을 닫음.

if __name__ == "__main__": 
    main() # 메인 함수를 시작함.
```

![image]()