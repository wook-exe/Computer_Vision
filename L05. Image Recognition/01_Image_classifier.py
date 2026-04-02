import tensorflow as tf # 인공신경망 모델 구축과 학습을 위해 텐서플로우 라이브러리를 임포트합니다.

def main(): # 프로그램의 메인 실행 흐름을 정의하는 함수를 선언합니다.
    mnist = tf.keras.datasets.mnist # 케라스 내장 데이터셋에서 MNIST 데이터셋 객체를 불러옵니다.
    (x_train, y_train), (x_test, y_test) = mnist.load_data() # 데이터셋을 훈련용(x_train, y_train)과 평가용(x_test, y_test) 튜플로 분할하여 다운로드 및 로드합니다.
    
    x_train = x_train / 255.0 # 훈련 이미지의 픽셀 값(0~255)을 255.0으로 나누어 0~1 사이의 값으로 정규화합니다.
    x_test = x_test / 255.0 # 테스트 이미지의 픽셀 값 역시 0~1 사이의 값으로 정규화하여 학습 조건과 맞춥니다.
    
    model = tf.keras.models.Sequential([ # 순차적으로 레이어를 쌓아 모델을 구성하기 위해 Sequential 객체를 생성합니다.
        tf.keras.layers.Flatten(input_shape=(28, 28)), # 28x28 크기의 2차원 이미지 배열을 784개의 1차원 배열로 평탄화하는 입력층을 추가합니다.
        tf.keras.layers.Dense(128, activation='relu'), # 128개의 노드를 가지며, 비선형성을 부여하기 위해 ReLU 활성화 함수를 사용하는 은닉층을 추가합니다.
        tf.keras.layers.Dense(10, activation='softmax') # 10개의 숫자(0~9) 각각에 대한 예측 확률을 출력하기 위해 Softmax 활성화 함수를 사용하는 출력층을 추가합니다.
    ]) # 모델 아키텍처 정의를 완료합니다.
    
    model.compile(optimizer='adam', # 오차를 최소화하기 위한 가중치 업데이트 알고리즘으로 효율적인 Adam 옵티마이저를 지정합니다.
                  loss='sparse_categorical_crossentropy', # 정수 형태의 라벨(y)을 원-핫 인코딩 없이 그대로 사용할 수 있는 다중 분류 손실 함수를 지정합니다.
                  metrics=['accuracy']) # 모델 훈련 및 평가 시 모니터링할 성능 지표로 정확도(accuracy)를 설정합니다.
    
    print("--- 모델 훈련 시작 ---") # 터미널에 학습 시작을 알리는 안내 문구를 출력합니다.
    model.fit(x_train, y_train, epochs=5) # 훈련 데이터셋을 사용하여 전체 데이터를 5번 반복(epochs=5)하며 모델 가중치를 학습시킵니다.
    
    print("--- 모델 평가 진행 ---") # 터미널에 평가 시작을 알리는 안내 문구를 출력합니다.
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2) # 학습에 사용되지 않은 테스트 데이터셋을 통해 모델의 최종 손실값과 정확도를 산출합니다.
    
    print(f"\n최종 테스트 정확도(Test Accuracy): {test_acc:.4f}") # 산출된 테스트 데이터의 최종 정확도를 소수점 4자리까지 포맷팅하여 터미널에 출력합니다.

if __name__ == '__main__': # 이 스크립트가 모듈로 임포트되지 않고 직접 실행될 때만 아래 코드를 실행하도록 합니다.
    main() # 위에서 정의한 main 함수를 호출하여 프로그램을 실행합니다.