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