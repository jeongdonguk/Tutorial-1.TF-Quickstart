# TF_Quickstart-for-beginners
이 레파지토리는 https://www.tensorflow.org/ 에서 제공하는 tutorial을 따라하며
텐서플로우에서 제공하는 딥러닝 코드에 대한 이해와 올바른 사용법을 익히기 위한 과정입니다.

# 실습해볼 내용
## 1.TensorFlow 2 quickstart for beginners
 : 케라스에서 제공하는 API를 활용하여 간단하게 모델을 구성하고 학습하는 과정을 실습해본다.
 ```python
 model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])
 ```
## 2.TensorFlow 2 quickstart for experts
 : 모델을 클래스화 하여 구성하는 방법에 대해 배워보고 만든 모델을 학습하는 과정을 실습해본다.
```python
class mymodel(Model):
    def __init__(self):
        super(mymodel, self).__init__() 
        # 2차원 conv레이어 정의
        self.conv1 = Conv2D(32, 3, activation='relu')
        # flatten 레이어 정의 : 1차원 텐서로 바꾸어주기 위함
        self.flatten = Flatten()
        # 위의 값을 dense레이어를 통과시켜 128가지로 나눈다.
        self.d1 = Dense(128, activation='relu')
        # 위의 값을 dense레이어를 통과시켜 10가지로 나눈다.
        self.d2 = Dense(10)
    
    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)

        return self.d2(x)


@tf.function
def train_step(images, labels):
    # 아래의 pred와 loss를 기록하기 위해 GradientTape를 사용
    with tf.GradientTape() as tape:
        pred = model(images, training=True)
        loss = loss_func(labels, pred)
    # loss 와 모델의 변수들을 통해 gradient를 계산한다.
    gradients = tape.gradient(loss, model.trainable_variables)
    # 계산된 gradient과 optimizer 함수를 통해 최적화된 모델의 변수들을 업데이트 해준다. 
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    train_loss(loss)
    train_accuracy(labels, pred)
```
#### 스터디를 위해 예제의 일부를 변형하거나 추가적인 코드를 사용될 수 있습니다.
이 레파지토리에서 실습해볼 예제의 출처는 아래와 같습니다 <br> https://www.tensorflow.org/tutorials
