import tensorflow as tf # 텐서플로우 라이브러리를 임포트하여 딥러닝 기능을 활성화합니다.
import numpy as np # 행렬 및 배열 연산, 그리고 argmax 등 수학적 처리를 위해 넘파이를 임포트합니다.
from tensorflow.keras.preprocessing import image # 로컬에 있는 테스트 이미지를 로드하고 처리하기 위한 모듈을 가져옵니다.

def main(): # 프로그램 실행의 진입점이 되는 메인 함수를 정의합니다.
    cifar10 = tf.keras.datasets.cifar10 # CIFAR-10 컬러 이미지 데이터셋을 불러올 수 있는 객체를 할당합니다.
    (x_train, y_train), (x_test, y_test) = cifar10.load_data() # 데이터를 다운로드하고 훈련 세트와 테스트 세트로 분할하여 변수에 저장합니다.
    
    x_train = x_train / 255.0 # 훈련 데이터의 픽셀 값을 0과 1 사이의 실수로 정규화하여 학습 효율과 안정성을 높입니다.
    x_test = x_test / 255.0 # 테스트 데이터도 훈련 데이터와 동일한 스케일을 가지도록 255.0으로 나누어 정규화합니다.
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', # CIFAR-10 데이터셋의 라벨(0~9)에 대응하는 실제 문자열 클래스 이름들을 리스트로 정의합니다.
                   'dog', 'frog', 'horse', 'ship', 'truck'] # 10개의 클래스 매핑용 리스트 정의를 완료합니다.

    model = tf.keras.models.Sequential([ # 레이어를 순서대로 쌓아 올리기 위해 Sequential 모델 아키텍처를 초기화합니다.
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)), # 32개의 3x3 필터를 사용하여 이미지의 특징을 추출하고 공간 차원을 유지하며 ReLU를 적용하는 첫 번째 합성곱 층입니다.
        tf.keras.layers.MaxPooling2D((2, 2)), # 2x2 영역에서 가장 큰 값만 남겨 이미지 크기를 절반으로 줄이고 핵심 특징을 요약하는 풀링 층입니다.
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'), # 64개의 3x3 필터를 사용하여 이전 층보다 더 복잡하고 추상적인 특징을 추출하는 두 번째 합성곱 층입니다.
        tf.keras.layers.MaxPooling2D((2, 2)), # 다시 한번 크기를 반으로 줄여 연산량을 감소시키고 공간적 변화에 대한 불변성을 부여합니다.
        tf.keras.layers.Flatten(), # 3차원 형태의 다채널 특징맵을 완전 연결 계층(Dense)에 넣기 위해 1차원 배열로 쭉 폅니다.
        tf.keras.layers.Dense(64, activation='relu'), # 64개의 뉴런을 가지는 은닉층으로, 추출된 특징들을 바탕으로 분류 패턴을 학습합니다.
        tf.keras.layers.Dense(10, activation='softmax') # 최종적으로 10개의 클래스에 속할 확률 분포를 출력하기 위해 Softmax 활성화 함수를 사용하는 출력층입니다.
    ]) # 전체 CNN 모델의 구조 정의를 마칩니다.
    
    model.compile(optimizer='adam', # 모델 갱신 시 경사하강법의 스텝 사이즈를 동적으로 조절하는 Adam 최적화 기법을 사용합니다.
                  loss='sparse_categorical_crossentropy', # 클래스가 원-핫 인코딩되지 않은 정수형 라벨(0~9)일 때 사용하는 손실 함수입니다.
                  metrics=['accuracy']) # 학습 중 모델의 정확도를 평가 지표로 추적 및 출력하도록 설정합니다.
    
    print("--- CNN 모델 훈련 시작 ---") # 터미널 창에 훈련 과정의 시작을 알리는 문자열을 출력합니다.
    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test)) # 훈련 데이터로 총 10 에포크 학습을 진행하며, 각 에포크마다 테스트 데이터로 검증(validation) 성능을 확인합니다.
    
    print("\n--- 모델 평가 진행 ---") # 학습이 끝난 후 본격적인 최종 평가가 시작됨을 알립니다.
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2) # 테스트 세트로 최종 손실과 정확도를 계산하여 변수에 담습니다.
    print(f"최종 테스트 정확도(Test Accuracy): {test_acc:.4f}") # 평가된 최종 정확도를 보기 쉽게 소수점 4자리까지 출력합니다.
    
    print("\n--- 외부 테스트 이미지 (dog.jpg) 예측 진행 ---") # 과제 요구사항인 개별 파일(dog.jpg) 예측 단계임을 터미널에 표시합니다.
    img_path = 'dog.jpg' # 예측 대상이 될 이미지 파일의 로컬 경로를 지정합니다. (동일 디렉토리에 위치해야 함)
    try: # 파일이 없거나 읽을 수 없는 오류를 방지하기 위해 예외 처리(Try-Except) 블록을 시작합니다.
        img = image.load_img(img_path, target_size=(32, 32)) # CIFAR-10 모델의 입력 규격에 맞게 타겟 사이즈를 32x32로 지정하여 이미지를 불러옵니다.
        img_array = image.img_to_array(img) # 불러온 PIL 이미지 객체를 다차원 넘파이 배열로 변환합니다.
        img_array = img_array / 255.0 # 학습 시 수행했던 전처리와 동일하게 픽셀 값을 0~1 사이로 정규화합니다.
        img_array = tf.expand_dims(img_array, 0) # 모델은 배치(Batch) 단위 입력을 기대하므로, 3차원 배열 앞부분에 1차원(배치 사이즈 1)을 추가하여 (1, 32, 32, 3) 형태로 만듭니다.
        
        predictions = model.predict(img_array) # 전처리된 단일 이미지를 훈련된 모델에 통과시켜 각 클래스에 대한 예측 확률값을 반환받습니다.
        predicted_index = np.argmax(predictions[0]) # 10개의 예측 확률 값 중 가장 큰 확률을 가진 인덱스 번호를 추출합니다.
        predicted_class = class_names[predicted_index] # 추출한 인덱스를 앞서 정의한 문자열 클래스 이름 리스트에 매핑하여 최종 클래스명을 얻습니다.
        
        print(f"예측 결과: 해당 이미지는 '{predicted_class}' (으)로 예측되었습니다.") # 모델이 추론한 최종 정답 텍스트를 터미널에 출력합니다.
    except FileNotFoundError: # 만약 지정한 경로에 'dog.jpg' 파일이 없다면 예외를 잡아냅니다.
        print(f"에러: '{img_path}' 파일을 찾을 수 없습니다. 스크립트와 동일한 폴더에 이미지를 추가해주세요.") # 파일 부재에 대한 안내 메시지를 출력하여 에러 원인을 알립니다.

if __name__ == '__main__': # 파이썬 인터프리터가 이 스크립트를 메인 프로그램으로 실행하는 경우에만 작동하도록 확인합니다.
    main() # 캡슐화된 메인 함수를 실행하여 전체 프로세스를 구동합니다.