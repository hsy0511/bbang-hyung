# bbang-hyung
# 1강 머신러닝 데이터셋
## colab 노트북 복사하는 방법
![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/e31a1036-24b6-4e7d-9812-adceb1882863)
## 머신러닝 데이터셋
![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/60898289-d4fe-4ae5-8f7a-de492aaac9ba)
## scikit-learn 소개
![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/b17fa33b-ef56-4871-a5ef-e53b583a9006)
## Iris 붓꽃 데이터셋
![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/4f7dcd4b-4f22-42f9-85e9-79905d548531)

데이터 셋 3개가 주어진다.
## Iris 데이터셋 로드하고 살펴보기
![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/a0fe78de-9ab4-4f6b-90ad-c1bd6a2629f1)

iris 꽃에 key 값을 딕셔너리 형태로 나타내어 살펴본다.
## features 특징 (=x)
![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/2dbe9ed3-8b8a-487a-b5de-8365276fca22)

함수 y = f(x)에 해당하는 특징 x 값을 나타낸다.
## target 정답 (=y)
![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/010fbf5f-41d4-4568-b774-4b44eff609d3)

함수 y = f(x)에 해당하는 정답 y값을 나타낸다.
## dataframe 변환
![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/642b0ea5-63bf-47b4-8979-ce6015eae11b)

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/6eaf17b9-a420-479e-b6f8-33334f33647c)

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/ddeea02c-7d56-4525-a767-bdb8acc171e6)

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/e38a6596-d79c-4d5d-bd0c-926ff9b28307)

판다스 패키지를 사용하여 데이터 셋들을 데이터 프레임 형태로 변환한다.
## 데이터셋 시각화
![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/d8c16fc2-262e-4169-aa36-4a4194bdeb12)
## 꽃받침의 길이와 너비에 따른 라벨 분포
![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/efcfa465-ecef-4e92-9bf8-5b11a2c85d1a)

꽃 마다 꽃받침의 길이, 너비를 그래프로 나타낸다.

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/c763e674-c2cc-47bd-b901-a9bd9d2c0adc)
## 꽃잎의 길이와 너비에 따른 라벨 분포
![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/135a29df-4de8-46b5-a4b4-f998b25409b6)

꽃 마다 꽃잎의 길이, 너비를 그래프로 나타낸다.
## 라벨 분포 그래프
![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/e209c6c9-fd35-4ad7-9041-f935fd1d1a8d)

타겟에 따른 카운트의 갯수로 그래프로 나타낸다.
## 꽃받침 길이의 분포 그래프
![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/fa206313-4f37-4038-b172-cf8b6c10b434)

꽃받침 길이에 따른 카운트를 그래프로 나타낸다
## 꽃잎 너비의 분포 그래프
![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/76e05864-efed-4972-88e2-d6056703f556)

꽃잎 너비 길이에 따른 카운트를 그래프로 나타낸다
## 꽃받침 길이 분포 그래프
![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/fe072bec-b2cf-434f-bb73-e5724abf9c1e)

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/23fb5b78-4c1c-4733-b95c-5e4cf6c18af2)

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/dc902dab-365d-4ee0-9367-3204d0fa6490)

박스 그래프에서 선에 가까울 수록 분포가 제일 많이 된 것이다. 검정색 다이아몬드는 데이터 분포에서 어긋난 데이터이다.
## 꽃잎 너비의 분포 그래프
![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/71b0f3fe-732c-4986-8203-4357bdd50085)

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/48c67256-95fc-4fad-b486-5db32e0610a8)

꽃의 대한 꽃잎 너비 길이를 그래프로 나타낸다
## [심화] 상관관계 correlation
![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/8f1ebae2-a40d-4b98-bcef-bd6335d95b10)

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/33adcb3c-200d-4e15-aa0c-9d7d27472f51)

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/14663d40-aea2-44b4-81a5-6a04f0f7f1f4)

그래프에서 x와y가 1:1 비율로 증가할 수록 좋은 상관관계인 것이다.
## 상관관계 매트릭스 correlation matrix
![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/7a11e0f5-8789-4f80-88cb-0a1f3c458a7e)

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/f7c944d8-efed-4725-bf93-73d89042c658)

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/e9178c92-64d0-4417-add1-f1ac1590b5d0)

매트릭스에서 1을 제외하고 1에 가장 근접하는 숫자가 상관관계가 좋다.
## 데이터셋의 분할
![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/cb4fc5bf-a209-4411-84e3-f166c8cea5c9)

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/c08fb831-0d3b-4bc5-8187-0e9298ee29a0)

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/271be64f-75c0-4b85-a81e-d7107155d2bf)

데이터프레임에서 가장 상위에 위치한 데이터 10개를 나타낸다

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/99eb3bc8-39d4-492b-ae3d-dba21422f642)

train_test_split 함수를 이용하여 훈련세와 테스트 세트로 데이터를 나눈다.

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/8efc191e-d59a-4bfa-a85e-1349552044cd)

x 훈련데이터에서 가장 상위 5개의 데이터를 나타낸다.

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/5bba2ebf-f151-4913-87dd-1cd60f462eb3)

y 훈련데이터에서 가장 상위 5개의 데이터를 나타낸다.

# 2강 분류
## 데이터셋 전처리
![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/47172187-2556-47d8-ac69-547831f796c7)

데이터를 이쁘게 정리하고 필터링을 해주는 역할을 한다. 머신러닝에 작업시간 80%는 전처리에 사용된다.

### 전처리가 필요한 이유
- 예시1
```
A의 집은 30평이고, B의 집은 지하철과의 거리가 3km이니까 A의 집이 더 비싸다.
```
위 예시는 비교하는 단위가 달라 직접적으로 비교가 불가능해서 비교는 틀렸다고 볼 수 있다.
- 예시2
```
A는 수학 시험에서 100점 만점에 50점을 받았다.
B는 수학 시험에서 500점 만점에 50점을 받았다.
A, B 모두 50점을 받았으므로 수학 실력이 비슷하다.
```
위 예시는 비교하는 단위가 같지만 비교하는 값의 범위가 달라 비교는 틀렸다고 볼 수 있다.

이러한 비교를 할 때 전처리 작업이 필요한 이유이다.
## 정규화 Normalization
정규화는 데이터를 0과 1사이의 범위를 가지도록 만듭니다. 같은 특성의 데이터 중에서 가장 작은 값을 0으로 만들고, 가장 큰 값을 1로 만들죠.

- 정규화 수식

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/e0a2a836-0074-4480-b954-eaeff9d6eac7)

- 정규화 예시
```
A는 수학 시험에서 100점 만점에 50점을 받았다.
B는 수학 시험에서 500점 만점에 50점을 받았다.
누가 수학을 더 잘하나?

A의 max 값 : 100
A의 min 값 : 0
A의 점수 : 50

B의 max 값 : 500
B의 min 값 : 0 
B의 점수 : 50
```
A는 (50 - 0)/(100 - )이 되어서 값은 0.5이고, 

B는 (50 - 0)/(500 - 0)이 되어서 값은 0.1이다.

즉 0.5 > 0.1이기 때문에 A가 수학을 더 잘할 가능성이 있는 것이다.

## Iris 데이터셋에서 정규화

```python 
from sklearn.preprocessing import MinMaxScaler
// 사이키런 라이브러리에서 MinMaxScaler 패키지 가져오기

scaler = MinMaxScaler()
// 클래스 초기화 및 scaler 객체 생성

scaled = scaler.fit_transform(x_train)
// x_train 데이터 값 정규화

scaled[:5]
// scaled에 데이터 배열 0부터 4까지 뽑아냄 
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/1dd5c1d9-302b-41cc-99d9-181016e31453)

데이터 프레임으로 나타낸다.

```python
x_train_scaled = pd.DataFrame(data=scaled, columns=iris['feature_names'])
// scaled 데이터를 열 에 라벨이 나오게 데이터 프레임으로 변경한다.

x_train_scaled.head()
// x_train_scaled 데이터 프레임에서 상위 5개 데이터를 뽑아냄
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/e9b506f0-cb90-4cc7-be23-ff031d476fa9)

- 박스 차트를 이용해서 전처리를 안한 데이터와 전처리를 한 데이터를 비교
```python
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 8))
// subplots 함수를 이용해서 사이즈가 8x8인 차트를 1행 2열로 만든다.

sns.boxplot(y=x_train['sepal length (cm)'], ax=axes[0])
// 전처리가 되지않은 박스차트를 0번째 열에 만든다.

sns.boxplot(y=x_train_scaled['sepal length (cm)'], ax=axes[1])
// 전처리가 된 박스차트를 1번째 열에 만는다.

plt.show()
// 차트를 보이게 한다.
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/21fe2233-43ec-418b-801c-de3c7b95f670)

박스 차트를 만드므로써 값의 범위가 바뀐것을 시각적으로 확인할 수 있다.
## 표준화 Standardization
표준화는 데이터의 분포를 정규분포로 바꿔줍니다. 즉 데이터의 평균이 0이 되도록하고 표준편차가 1이 되도록 만들어준다.

- 표준화 수식
![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/3929272c-ebdf-4e17-b56a-b233d93ffe79)

표준화는 데이터의 평균을 0으로 만들어주면 데이터의 중심이 0에 맞춰지게 되고, 표준편차를 1로 만들어 주면 데이터가 예쁘게 정규화가 된다. 이렇게 표준화를 시키게 되면 일반적으로 학습속도가 빠르고 정확도가 높아진다.

- 예시
```
중심 0 : Zero-centered
정규화 : Normalized
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/bfa22fd1-e992-40ef-b4f9-ca54cef51a41)

## Iris 데이터셋에서 표준화
```python
from sklearn.preprocessing import StandardScaler
// 사이키런 라이브러리에서 StandardScaler 패키지 가져오기

scaler = StandardScaler()
// 클래스 초기화 및 scaler 객체 생성

scaled = scaler.fit_transform(x_train)
// x_train 데이터 값 표준화

scaled[:5]
// scaled에 데이터 배열 0부터 4까지 뽑아냄 
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/78163a12-3794-4572-a67f-e27f81cbb498)

데이터 프레임으로 변환

```python
x_train_scaled = pd.DataFrame(data=scaled, columns=iris['feature_names'])
// scaled 데이터를 열 에 라벨이 나오게 데이터 프레임으로 변경한다.

x_train_scaled.head()
// x_train_scaled 데이터 프레임에서 상위 5개 데이터를 뽑아냄
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/d68f3835-cafa-4f3c-8e91-c7dbf05174f7)

- 박스 차트를 이용해서 전처리를 안한 데이터와 전처리를 한 데이터를 비교
```python
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 8))
// subplots 함수를 이용해서 사이즈가 8x8인 차트를 1행 2열로 만든다.

sns.boxplot(y=x_train['sepal length (cm)'], ax=axes[0])
// 표준화가 되지않은 박스차트를 0번째 열에 만든다.

sns.boxplot(y=x_train_scaled['sepal length (cm)'], ax=axes[1])
// 표준화가 된 박스차트를 1번째 열에 만는다.

plt.show()
// 차트를 보이게 한다.
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/70daa4a8-77dc-47db-92d2-20fe39631f2e)

박스 차트를 만드므로써 값의 범위가 바뀐것을 시각적으로 확인할 수 있다.
## 인코딩 Encoding
일반적인 머신러닝 알고리즘은 숫자만을 입력으로 받을 수 있다. 왜냐하면 모든 계산을 수식으로 하기 때문이다.

따라서 사용할 모든 데이터를 숫자로 변환하는 작업이 필요하다.
### 펭귄 데이터셋
부리의 길이와 부리의 높이로 이루어짐

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/b3132b64-4831-45b6-bc4e-07ce5be650b1)

- 데이터 프레임으로 펭귄 데이터 나타내기
```python
penguins = sns.load_dataset('penguins')
// 시본 라이브러리에 loal_dataset 함수를 통해서 펭귄데이터를 데이터 프레임으로 나타낸다.

penguins.head()
// 펭귄 데이터 프레임에서 상위 5개 데이터를 뽑아냄
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/6d1f0380-a98c-4f25-9d0f-e3a9ce397bb8)

## Null, NaN 삭제
```python
penguins = penguins.dropna()
// dropna 함수를 사용하여 결측값(null, nan)이 들어있는 행 전체를 제거함

penguins.head()
// 펭귄 데이터 프레임에서 상위 5개 데이터를 뽑아냄
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/5029292a-1bf3-402f-81be-0689c25d088c)

```python
plt.figure(figsize=(16, 8))
// 사이즈가 16x8인 차트를 만든다.

sns.countplot(x=penguins['species'])
// x축이 펭귄의 종류이고 종류마다 펭귄에 마릿수를 세주는 차트를 생성한다. 

plt.show()
// 차트를 보여준다.
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/aad3d1a3-59ba-4f03-a0c3-13d77a9a8841)

- 펭귄의 종류를 숫자로 변경

```python
from sklearn.preprocessing import LabelEncoder
// 사이키런 라이브러리에서 LabelEncoder 패키지 가져오기

encoder = LabelEncoder()
// 클래스 초기화 및 encoder 객체 생성

encoded = encoder.fit_transform(penguins['species'])
// encoder 값 수치화(문자를 숫자로 변경)

encoded
// encoded 값 나타냄
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/f0dd44a9-80af-4c34-8b2c-db4154ea393b)

종류별로 0,1,2로 수치화 된것을 볼 수 있음.

- 수치화한 데이터 프레임 확인

```python
penguins['species'] = encoded
// 펭귄의 종류를 encoded(수치화 한 값)으로 나타냄

penguins.head()
// 펭귄 데이터 프레임에서 상위 5개 데이터 뽑아냄
```

- 수치화 한 데이터 차트로 확인

```python
plt.figure(figsize=(16, 8))
// 16x8짜리 차트 생성

sns.countplot(x=penguins['species'])
// x축이 펭귄의 종류이고 종류마다 펭귄에 마릿수를 세주는 차트를 생성한다.

plt.show()
// 차트를 보여준다.
```

- 수치화 된 데이터 라벨 값 확인하기

```python
encoder.classes_
// encoder 데이터 값에 등록된 class를 확인한다. 
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/a0b156c6-f6d9-41e9-80cf-70bfe62c69af)

- 수치화 된 값 반대로 확인하기

```python
inversed = encoder.inverse_transform(encoded)
// encoder 데이터 값을 반대로 확인한다.

inversed
// inversed를 나타냄

```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/0efdca35-eda4-42e5-854d-293f61d20770)

## 간단한 분류기의 예

## SVM 실습

### SVM 실습 1. Iris 데이터셋을 이용한 실습
- 데이터셋 로드
- 전처리
- 모델정의
- 학습 Training
- 검증 Validation

### SVM 실습 2. MANIST 데이터셋을 이용한 실습
- 데이터셋 로드
- 데이터 시각화
- 데이터 전처리 - 정규화
- 데이터 전처리 - 데이터셋 분할
- 학습, 검증
- 검증 결과 시각화
## KNN 실습

## 샘플 데이터로 KNN 이해하기

## MNIST 데이터셋을 이용한 실습
- 모델 정의, 학습
- 검증
- 최적의 k 검색
## 연습해볼만한 데이터셋 소개
### Scikit-learn Sample Datasets
https://scikit-learn.org/stable/datasets/toy_dataset.html

- load_boston(*[, return_X_y])
  - Load and return the boston house-prices dataset (regression).

- load_iris(*[, return_X_y, as_frame])
  - Load and return the iris dataset (classification).

- load_diabetes(*[, return_X_y, as_frame])
  - Load and return the diabetes dataset (regression).

- load_digits(*[, n_class, return_X_y, as_frame])
  - Load and return the digits dataset (classification).

- load_linnerud(*[, return_X_y, as_frame])
  - Load and return the physical excercise linnerud dataset.

- load_wine(*[, return_X_y, as_frame])
  - Load and return the wine dataset (classification).

- load_breast_cancer(*[, return_X_y, as_frame])
  - Load and return the breast cancer wisconsin dataset (classification).
### Seaborn Sample Datasets
https://github.com/mwaskom/seaborn-data

- car_crashes: https://www.kaggle.com/fivethirtyeight/fivethirtyeight-bad-drivers-dataset
- dots: https://shadlenlab.columbia.edu/resources/RoitmanDataCode.html
- fmri: https://github.com/mwaskom/Waskom_CerebCortex_2017
- penguins: https://github.com/allisonhorst/penguins
- planets: https://exoplanets.nasa.gov/exoplanet-catalog/
# 3강 회귀

# 4강 논리 회귀와 의사결정나무

# 5강 비지도학습

# 6강 머신러닝 실전 스킬
