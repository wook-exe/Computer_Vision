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