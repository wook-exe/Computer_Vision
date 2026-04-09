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
    # YOLOv3 모델 가중치와 설정 파일 경로를 지정함.
    weights_path = "yolov3.weights" # YOLOv3 가중치 파일 경로를 저장함.
    config_path = "yolov3.cfg" # YOLOv3 네트워크 설정 파일 경로를 저장함.
    
    # OpenCV DNN 모듈을 사용하여 YOLO 모델을 로드함.
    net = cv2.dnn.readNet(weights_path, config_path) # 네트워크 객체를 생성함.
    layer_names = net.getLayerNames() # 모든 레이어 이름을 가져옴.
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()] # 출력 레이어 추출함.
    
    cap = cv2.VideoCapture("slow_traffic_small.mp4") # 입력 비디오 파일을 오픈함.
    tracker = Sort(max_age=1, min_hits=3, iou_threshold=0.3) # SORT 추적기를 초기화함.
    
    while cap.isOpened(): # 비디오 재생 동안 루프를 실행함.
        ret, frame = cap.read() # 프레임을 읽어옴.
        if not ret: break # 읽기 실패 시 종료함.
            
        height, width, _ = frame.shape # 프레임 크기를 추출함.
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False) # Blob 변환함.
        net.setInput(blob) # 네트워크 입력 설정함.
        outs = net.forward(output_layers) # 순방향 추론 실행함.
        
        boxes, confidences = [], [] # 박스와 신뢰도 저장 리스트임.
        for out in outs: 
            for detection in out:
                scores = detection[5:] 
                class_id = np.argmax(scores) 
                confidence = scores[class_id]
                if confidence > 0.5: # 신뢰도 0.5 이상만 필터링함.
                    w, h = int(detection[2] * width), int(detection[3] * height)
                    x, y = int(detection[0] * width - w / 2), int(detection[1] * height - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4) # NMS 적용함.
        dets = [] 
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                dets.append([x, y, x + w, y + h, confidences[i]]) # SORT 입력 규격 가공함.
                
        dets = np.array(dets) if len(dets) > 0 else np.empty((0, 5)) # Numpy 배열 변환함.
        trackers = tracker.update(dets) # 추적 정보 업데이트함.
        
        for trk in trackers:
            x1, y1, x2, y2, obj_id = [int(v) for v in trk]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2) # 추적 박스 렌더링함.
            cv2.putText(frame, f"ID: {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2) # ID 출력함.
            
        cv2.imshow("Multi-Object Tracking", frame) # 결과 표시함.
        if cv2.waitKey(1) & 0xFF == 27: break # ESC 키 입력 시 종료함.
            
    cap.release() # 자원 해제함.
    cv2.destroyAllWindows() # 윈도우 닫음.

if __name__ == "__main__": main() # 메인 함수 실행함.
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
import cv2 # OpenCV 라이브러리를 임포트하여 실시간 영상 처리를 수행함.
import mediapipe as mp # Mediapipe 라이브러리를 임포트하여 얼굴 랜드마크 기능을 사용함.

def main(): # 메인 실행 함수를 정의함.
    mp_face_mesh = mp.solutions.face_mesh # FaceMesh 모듈을 변수에 할당함.
    cap = cv2.VideoCapture(0) # 기본 웹캠 스트림을 오픈함.
    
    with mp_face_mesh.FaceMesh( 
        static_image_mode=False, # 비디오 스트림 모드를 활성화함.
        max_num_faces=1, # 탐지 대상을 얼굴 1개로 제한함.
        refine_landmarks=False, # PDF 요구사항에 맞춰 기본 468개 포인트만 추출함.
        min_detection_confidence=0.5, # 검출 최소 신뢰도를 설정함.
        min_tracking_confidence=0.5 # 추적 최소 신뢰도를 설정함.
    ) as face_mesh: 
    
        while cap.isOpened(): 
            success, image = cap.read() # 프레임을 읽어옴.
            if not success: continue 
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # RGB 변환함.
            image_rgb.flags.writeable = False # 성능 최적화를 위해 읽기 전용 설정함.
            results = face_mesh.process(image_rgb) # 랜드마크 추출 연산을 수행함.
            
            image_rgb.flags.writeable = True # 쓰기 권한을 복구함.
            ih, iw, _ = image.shape # 이미지의 실제 픽셀 크기를 가져옴.
            
            if results.multi_face_landmarks: # 랜드마크 검출 시 내부를 실행함.
                for face_landmarks in results.multi_face_landmarks: 
                    for landmark in face_landmarks.landmark: 
                        x, y = int(landmark.x * iw), int(landmark.y * ih) # 픽셀 좌표로 변환함.
                        cv2.circle(image, (x, y), 1, (0, 255, 0), -1) # 초록색 점을 그림.
                        
            cv2.imshow('Mediapipe FaceMesh', cv2.flip(image, 1)) # 거울 모드로 화면에 표시함.
            if cv2.waitKey(5) & 0xFF == 27: break # ESC 키 입력 시 종료함.
                
    cap.release() # 자원을 반환함.
    cv2.destroyAllWindows() # 모든 창을 닫음.

if __name__ == "__main__": main() # 메인 함수를 시작함.
```

![image]()