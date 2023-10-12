# bbang-hyung

https://github.com/hsy0511/bbang-hyung/blob/main/README.md#%EB%AA%A8%EB%8D%B8-%EC%A0%95%EC%9D%98-%ED%95%99%EC%8A%B5-%EA%B2%80%EC%A6%9D

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

- 정규화 데이터 프레임으로 변환

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

- 표준화 데이터 프레임으로 변환

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

# 2강 분류
## 간단한 분류기의 예
### 분류기란?
강아지와 고양이의 사진을 보고 분유하는 기계

- 분류 : Classify
- 기계 : machine
- 분류기 : classifier
## 강아지와 고양이를 구분하는 분류기
![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/9e0a486b-1ba5-4077-a5bf-cbac5967e157)
## 어떻게 선을 그려야 잘 구분할 수 있을까?
구분선과 강아지, 고양이 사이의 거리를 구하여 거리가 최대가 되는 선을 긋는다.

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/2ee4a109-77e4-45fa-a4cd-2fe1fa455106)

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/5504e1e9-b970-4864-be14-16812bdd2e12)

## SVM 실습

### SVM 실습 1. Iris 데이터셋을 이용한 실습
- 데이터셋 로드

```python
from sklearn.datasets import load_iris
// 사이킷런 라이브러리에서 load_iris 패키지를 가져온다.

import pandas as pd
// 판다스 패키지를 가져온다.

iris = load_iris()
// 클래스 초기화 및 iris 객체 생성

df = pd.DataFrame(data=iris['data'], columns=iris['feature_names'])
// iris 데이터에대한 데이터 프레임 생성

df['target'] = iris['target']
// 데이터 프레임의 타겟을 iris 타겟으로 변경

df.head()
// df 데이터 프레임에서 상위 5개 데이터 나타냄.

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/6cf86586-3e12-48f6-8035-e98864bebb33)

```

- 전처리
```python
from sklearn.model_selection import train_test_split
// 사이킷런 라이브러리에서 train_test_split 패키리를 가져온다.

x_train, x_val, y_train, y_val = train_test_split(df.drop(columns=['target']), df[['target']], test_size=0.2, random_state=2021)
// train_test_split으로 x에서는 타겟을 빼고 넣고 y에는 타겟만 넣고 Validation의 값은 20% train셋 크기는 80%로 랜덤으로 나눈다.

print(x_train.shape, y_train.shape)
// 몇개씩 나뉘었는지 shape를 사용하여 알아본다.

print(x_val.shape, y_val.shape)
// 몇개씩 나뉘었는지 shape를 사용하여 알아본다.
```
![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/6c0c2504-3883-40e0-8bc1-267fbc2e6678)

- 모델정의
```python
from sklearn.svm import SVC
// 사이킷런 라이브러리에서 svc(support vector classifier) 패키지를 가져온다

model = SVC()
// svc에 대한 모델을 정의한다
```
- 학습 Training
```python
model.fit(x_train, y_train['target'])
// 모델에 x데이터와 y데이터를 넣어서 훈련시킨다.
// 정답을 알려주어서 훈련시키는 이유는 지도 학습이기 때문이다.
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/3fe778a9-f39b-4ac1-b61a-5dc4a6ff64b9)

- 검증 Validation
```python
from sklearn.metrics import accuracy_score
// 사이킷런 라이브러리에서 accuracy_score(정확도 점수) 패키지를 가져온다.

y_pred = model.predict(x_val)
// x_val 데이터를 예측한다.

accuracy_score(y_val, y_pred) * 100
// 예측한 데이터를 정답 값과 비교하여 정확도를 확인한다.
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/9525f6c7-5f8d-4c1f-a523-3a5ee7a95ec3)

### SVM 실습 2. MANIST 데이터셋을 이용한 실습
손글씨 숫자 이미지를 보고 0-9를 분류하는 분류기

- 데이터셋 로드
```python
from sklearn.datasets import load_digits
// 사이킷런 라이브러리에서 load_digits 패키지를 가젼온다.

digits = load_digits()
// 클래스 초기화 및 digits 객체 생성

digits.keys()
//  digits 딕셔너리의 key 값을 나타낸다.
```
![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/d2a45f4c-ad43-48b4-b6c0-bd548f436969)

- 데이터 시각화

```python
data = digits['data']
// data변수에 데이터를 저장한다.

data.shape
// data가 어떤 형태로 이루어져 있는지 확인
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/18b1572b-8d1a-433b-b106-10dfa8a07349)

```python
target = digits['target']
// target 변수에 타겟 데이터를 저장한다.

target.shape
// target가 어떤 형태로 이루어져 있는지 확인
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/64ebbe1f-8a71-4caf-92d2-94fc0f5a1e3e)

데이터는 1797개가 있다.

```python
target = digits['target']
// target 변수에 타겟 데이터를 저장한다.

target.shape
// target이 어떤 형태로 이루어져 있는지 확인
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/696e3425-8636-495f-9195-14fe5e990565)

타겟 값도 1797개가 있다. x값과 라벨값에 크기가 같다.
```python
import matplotlib.pyplot as plt
// matplotlib.pyplot 패키지를 가져온다

fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(16, 8))
// 2행5열 16x8 사이즈에 차트를 만든다.

for i, ax in enumerate(axes.flatten()):
// flatten를 사용하여 축을 1차원으로 만든다.

  ax.imshow(data[i].reshape((8, 8)))
// 데이터에 i번째를 하나씩 다 뽑아오는데 reshape로 8x8 크기로 뽑아온다. 

  ax.set_title(target[i])
// 이 차트에 제목은 차겟에 i번째로 라벨을 지정한다.
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/7ea78aaa-c9a4-4257-9f59-258d38bba0e9)

- 데이터 전처리 - 정규화
```python
data[0]
// 데이터에 0번째 인덱스를 뽑아온다.
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/5c627b4a-2eff-4262-a863-943cf921e48f)

```python
from sklearn.preprocessing import MinMaxScaler
// 사이킷런 패키지에서 MinMaxScaler 패키지를 가져온다

scaler = MinMaxScaler()
// 클래스 초기화 scaler 객체생성

scaled = scaler.fit_transform(data)
// data를 정규화시킨다.

scaled[0]
// scaled에 0번째 인덱스를 뽑아온다.
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/8ced4e65-56fe-412d-93c2-9a4e53a1b706)

- 데이터 전처리 - 데이터셋 분할
```python
from sklearn.model_selection import train_test_split
// 사이킷런 라이브러리에서 train_test_split 패키지를 가져온다.

x_train, x_val, y_train, y_val = train_test_split(scaled, target, test_size=0.2, random_state=2021)
// train_test_split를 사용하여 x에는 정규화된 데이터를 넣어주고 y에는 타겟을 넣어주고 val은 20%, train은 80%로 랜덤으로 나눈다.

print(x_train.shape, y_train.shape)
// 몇개씩 나뉘었는지 shape를 사용하여 알아본다.

print(x_val.shape, y_val.shape)
// 몇개씩 나뉘었는지 shape를 사용하여 알아본다.
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/f139166c-33b4-4613-8e0f-ff687b79869f)

- 학습, 검증
```python
from sklearn.svm import SVC
// 사이킷런 라이브러리에서 SVC 패키지를 가져온다.

from sklearn.metrics import accuracy_score
// 사이킷런 라이브러리에서 accuracy_score 패키지를 가져온다.

model = SVC()
// svc에 대한 모델을 정의한다.

model.fit(x_train, y_train)
// 모델을 훈련시킨다.

y_pred = model.predict(x_val)
// x_val 데이터 값을 예측한다.

accuracy_score(y_val, y_pred) * 100
// 예측한 값과 정답 값과 비교하여 정확도를 채점한다.
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/0ac156ca-dfa7-4273-a1ed-6972d4d0b339)

- 검증 결과 시각화
```python
import matplotlib.pyplot as plt
// matplotlib.pyplot 패키지를 가져온다.

fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(16, 8))
// 2행5열 16x8 사이즈에 차트를 만든다.

for i, ax in enumerate(axes.flatten()):
// flatten를 사용하여 축을 1차원으로 만든다.

  ax.imshow(x_val[i].reshape((8, 8)))
// 데이터에 i번째를 하나씩 다 뽑아오는데 reshape로 8x8 크기로 뽑아온다.

  ax.set_title((f'True: {y_val[i]}, Pred: {y_pred[i]}'))
// 각 그림의 이름은 ture : 정답 값, pred : 예측한 값으로 나타낸다.
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/a7bd5f84-2380-4d8a-b327-3859d09e5e55)

틀린게 있으면 모델을 튜닝하여 정확도를 100%로 만들어야한다.

## KNN 실습
K-최근접 알고리즘

KNN은 비슷한 특성을 가진 개체끼리 나누는 알고리즘이다. 예를 들어 하얀 고양이가 새로 나타났을 때 일정 거리안에 다른 개체들의 개수(k)를 보고 자신의 위치를 결정한다.

- k = 2 일때 고양이 분류

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/9ea556b0-f27e-4864-837b-6d1997a369a8)

## 샘플 데이터로 KNN 이해하기
k = 3

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
import random
// KNeighborsClassifier, make_classification, random 패키지를 가져온다.
// make_classification 패키지는 분류 문제에 대한 가상의 데이터셋을 생성한다.

x, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=1)
// n_samples: 생성할 데이터 포인트의 총 수
// n_features: 각 데이터 포인트가 가질 특성 또는 열의 수
// n_informative: 클래스와 관련된 유용한 정보를 가진 특성의 수
// n_redundant: 클래스와 관련이 없는 중복 특성의 수
// n_classes: 생성할 클래스의 수
// random_state: 데이터를 생성하기 위한 무작위 시드

red = x[y == 0]
blue = x[y == 1]
// x 데이터에서 y가 0이면 red, 1이면 blue가 나오게 한다.

new_input = [[random.uniform(-2, 2), random.uniform(-2, 2)]]
// -2부터 2 사이의 실수 중에서 난수값을 리턴

plt.figure(figsize=(10, 10))
// 가로 10 세로 10짜리 차트를 만든다.

plt.scatter(x=red[:, 0], y=red[:, 1], c='r')
plt.scatter(x=blue[:, 0], y=blue[:, 1], c='b')
// red 데이터는 빨간색으로 나타내고 blue 데이터는 파란 데이터로 나타낸다.

model = KNeighborsClassifier(n_neighbors=3)
// 모델 정의

model.fit(x, y)
// 모델 훈련

pred = model.predict(new_input)
// 모델 값 예측

pred_label = 'red' if pred == 0 else 'blue'
// 모델 예측값이 0이면 red를 표기하고 아니면 blue를 표기한다.

plt.scatter(new_input[0][0], new_input[0][1], 100, 'g')
// new_input[0][0], new_input[0][1] 위치에 산점도 사이즈가 100인 초록색 점을 표기한다

plt.annotate(pred_label, xy=new_input[0], fontsize=16)
// plt.annotate : 주석 함수
// xy 위치에 pred_label 글자를 16 사이즈 폰트로 주석을 단다.

plt.show()
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/a4b602cf-7322-4d95-8cff-ddfea3c7cbbe)

계속 초록색에 타겟이 변경된다. k-초근접 이웃으로 타겟이 무슨 색인지 예측하여 알 수 있다.
## MNIST 데이터셋을 이용한 실습
- 모델 정의, 학습

```python
from sklearn.neighbors import KNeighborsClassifier
# KNeighborsClassifier 패키지를 가져온다

model = KNeighborsClassifier(n_neighbors=5)
# KNeighborsClassifier에 대한 모델을 정의한다.

model.fit(x_train, y_train)
# 모델을 훈련시킨다.
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/312d179e-4cbb-44bc-9897-502b377e45a8)

- 검증

```python
y_pred = model.predict(x_val)
# x_val의 데이터 값을 예측한다.

accuracy_score(y_val, y_pred) * 100
# 예측한 값과 결과 값을 비교하여 정확도를 채점한다.
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/89fcc8be-c14b-4b4e-8345-2fc0e8b53527)

- 최적의 k 검색

```python
for k in range(1, 11):
// 모델을 10번 학습시킨다.

  model = KNeighborsClassifier(n_neighbors=k)
// KNeighborsClassifier에 대한 모델을 정의한다.

  model.fit(x_train, y_train)
// 모델을 훈련시킨다.

  y_pred = model.predict(x_val)
// x_val의 데이터 값을 예측한다.

  print(f'k: {k}, accuracy: {accuracy_score(y_val, y_pred) * 100}')
// k:k, accuracy: 정답 값과 예측한 값을 비교했을 때 예측한 값에 정확도로 나타낸다.
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/2116425e-1a2f-4c76-9b15-073f10498073)


# 3강 회귀
### 목차
- 회귀 Regressor 소개
- 선형 회귀 Linear Regression 실습
## 분류와 회귀
### 강아지와 고양이 구분
강아지와 고양이의 중간은 없다.

### 부동산 가격 예측
부동산 가격은 연속적인 실수로 나타낼 수 있다.
- 1000만원
- 1001만원
- 1억 2500만원
- 51억 4123만원
## 선형회귀 실습
### 보스턴 집 값 데이터셋을 이용한 실습
#### 데이터셋 로드
- CRIM: 범죄율
- ZN: 25,000평방 피트 당 주거용 토지의 비율
- INDUS: 비소매 비즈니스 면적 비율
- CHAS: 찰스 강 더미 변수 (통로가 하천을 향하면 1; 그렇지 않으면 0)
- NOX: 산화 질소 농도 (1000만 분의 1)
- RM: 평균 방의 개수
- AGE: 1940년 이전에 건축된 자가 소유 점유 비율
- DIS: 5개의 보스턴 고용 센터까지의 가중 거리
- RAD: 고속도로 접근성 지수
- TAX: 10,000달러 당 전체 가치 재산 세율
- PTRATIO 도시별 학생-교사 비율
- B: 1000 (Bk-0.63) ^ 2 (Bk는 도시별 검정 비율)
- LSTAT: 인구의 낮은 지위
- target: 자가 주택의 중앙값 (1,000달러 단위)

```python
from sklearn.datasets import load_boston
import pandas as pd

data = load_boston()

df = pd.DataFrame(data['data'], columns=data['feature_names'])
df['target'] = data['target']

df.tail()
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/78b583c1-2847-4a19-955d-8c0e40e0a7a0)

#### 데이터 시각화
- Distribution plot

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.displot(x=df['target'])
plt.show()
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/a87fe7de-76ec-405e-8082-dcb302ad0290)

- correlation matrix

```python
plt.figure(figsize=(10, 10))
corr = df.corr()
sns.heatmap(corr, annot=True, square=True, cmap='PiYG', vmin=-1, vmax=1)
plt.show()
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/dd74eb7f-9f3b-4fbc-8cbc-c4b14c2e512f)

#### 데이터셋 분할
```python
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(df.drop(columns=['target']), df['target'], test_size=0.2, random_state=2021)

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/993287b4-69ea-433a-a7c4-0a802aabc1a8)

#### 모델 정의
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
```
#### 학습
```python
model.fit(x_train, y_train)
```
#### 검증
```python
x_val[:5]
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/20957ee3-d857-40d7-8166-6dd5f43520d4)

```python
y_pred = model.predict(x_val)

print(y_pred[:5])
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/1b945e6d-d6ac-41c7-843f-3a1dc488be1c)

#### 회귀에서의 정확도는 어떻게 구할 수 있을까?
- 정답값과 예측값이 차이가 작으면 정확도가 높다
- 정답값과 예측값의 차이가 크면 정확도가 낮다

```python
print(list(y_val[:5]))

print(list(y_pred[:5]))
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/6d1602bb-3e9e-461a-8043-fcaacf923cd6)

```python
y_val[:5] - y_pred[:5]
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/b7b97f4e-bb7e-43fa-b3be-b7faf1dbe941)

```python
abs(y_val[:5] - y_pred[:5])
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/e9322218-b204-44a9-9fc6-03a65ce4f9cd)

#### Mean Absolute Error
- MAE
- 정답과 예측값 차이의 절대값의 평균

```python
abs(y_val[:5] - y_pred[:5]).mean()
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/3a4d443b-f7e3-4f55-9b44-893f6b09ed81)

```python
abs(y_val - y_pred).mean()
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/7988b70b-9892-4fd8-bf3f-59c91c4ef58a)

```python
from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_val, y_pred)
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/9acba601-5c6d-4068-a9c6-2948058749c3)

#### Mean Squared Error
- MSE
- 정답과 예측값 차이의 제곱의 평균

```python
((y_val - y_pred) ** 2).mean()
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/b36b829f-11b1-4eb1-aa06-ff12fd7ac77b)

```python
from sklearn.metrics import mean_squared_error

mean_squared_error(y_val, y_pred)
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/bbb445e0-b02f-43d8-b387-00c17a5a2743)

#### 표준화
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.fit_transform(x_val)

print(x_train_scaled[:5])
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/98c6f379-4e65-4fd3-a7a2-48516aabfb3a)

```python
model = LinearRegression()

model.fit(x_train_scaled, y_train)

y_pred = model.predict(x_val_scaled)

mean_absolute_error(y_val, y_pred)
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/fcf51783-e3e1-4912-9ead-a6dd2992d4b7)

#### 정규화
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.fit_transform(x_val)

print(x_train_scaled[:5])
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/afd255f7-f357-4083-aff6-eacf52a5ed7f)

```python
model = LinearRegression()

model.fit(x_train_scaled, y_train)

y_pred = model.predict(x_val_scaled)

mean_absolute_error(y_val, y_pred)
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/3ae7ebb8-5ce6-40ae-8fa9-fb5ad9f2d737)

### Linear Regression (선형회귀)
![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/399937d7-e4ae-4e42-9a10-acbca1922e0e)

### Ridge Regression 맛보기
학습이 과대적합 되는 것을 방지하기위해 패널티를 부여한다. (L2 Regularazation)

용어를 몰라도 문서를 보고 다른 알고리즘을 사용하는 방법을 익혀보자.

- 링크 : linear_model의 모델들이 나온다.

https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/5665bd7b-6452-407e-b785-f7841cd1a047)

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/c0d86cd9-5aa7-4013-8b27-642b60918c35)

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/65beea58-112b-4c20-aa22-b5d6bbced63e)

```python
from sklearn.linear_model import Ridge

model = Ridge()

model.fit(x_train, y_train)

y_pred = model.predict(x_val)

mean_absolute_error(y_val, y_pred)
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/d8780000-77e7-4a8e-9b72-a1c6eddfc3f0)

### 당뇨병 데이터셋을 이용한 실습
- 링크 : 당뇨병에 대한 데이터셋이 나온다.

https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/c2b16cdd-ab7a-4740-a1ef-a9dd3ad1d112)

#### 데이터셋 로드
- age: 나이
- sex: 성별
- bmi: BMI 체질량지수
- bp: 평균 혈압
- s1 tc: 총 혈청 콜레스테롤
- s2 ldl: 저밀도 지방단백질
- s3 hdl: 고밀도 지방단백질
- s4 tch: 총 콜레스테롤 / HDL
- s5 ltg: 혈청 트리글리세리드 수치의 로그
- s6 glu: 혈당 수치
- target: 1년 후 당뇨병 진행도

```python
from sklearn.datasets import load_diabetes
import pandas as pd

data = load_diabetes()

df = pd.DataFrame(data['data'], columns=data['feature_names'])
df['target'] = data['target']

df.head()
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/379e1ede-1ec4-4182-80cc-6da83303beb8)

#### 데이터 시각화
- 이미 표준화가 되어 있는 데이터셋

```python
sns.boxplot(y=df['age'])
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/65ac2b9c-3719-4707-9d40-27e07c4fac27)

```python
sns.displot(x=df['target'])
plt.show()
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/c1c60702-d69c-4a27-bb97-ea6d8724a90a)

```python
plt.figure(figsize=(10, 10))
corr = df.corr()
sns.heatmap(corr, annot=True, square=True, cmap='PiYG', vmin=-1, vmax=1)
plt.show()
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/003922a6-c7c4-4e11-8c31-d234746392a9)

#### 데이터셋 분할
```python
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(df.drop(columns=['target']), df['target'], test_size=0.2, random_state=2021)

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/18921f4a-fb6a-4598-bee2-a638774f49fd)

#### 모델 정의, 학습, 검증
```python
from sklearn.linear_model import SGDRegressor

model = SGDRegressor()

model.fit(x_train, y_train)

y_pred = model.predict(x_val)

mean_absolute_error(y_val, y_pred)
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/1d26efe7-48c3-4120-9d08-f3e5e4413fa3)

```python
model = LinearRegression()

model.fit(x_train, y_train)

y_pred = model.predict(x_val)

mean_absolute_error(y_val, y_pred)
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/6eeb6162-f645-40dd-aa5a-d5377ce2205a)

#### 검증 결과 시각화
```python
plt.figure(figsize=(8, 6))
sns.scatterplot(x=x_val['bmi'], y=y_val, color='b')
sns.scatterplot(x=x_val['bmi'], y=y_pred, color='r')
plt.show()
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/b4870b65-7c81-4bf7-9e05-8632dbade10c)

# 4강 논리 회귀와 의사결정나무

# 5강 비지도학습

# 6강 머신러닝 실전 스킬
