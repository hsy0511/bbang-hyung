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
