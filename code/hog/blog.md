# Histogram of Gradients
이번에 할 HOG는 feature descriptor의 하나이다
<br>쉽게 말해서 이미지의 특징들을 뽑아내서 다른 방식으로
<br>표현하는 것이다
<br>HOG는 그중에서 Gradient에 집중하여 하는 것을 말한다

## 예상 결과물
사진
https://www.learnopencv.com/histogram-of-oriented-gradients/
<br>위의 사이트에서 OpenCv를 이용하여 진행한 HOG의 출력이다

하지만 Opencv를 사용하지 numpy만을 사용하여 진행하고
<br> feature descriptor를 완성하게 되면 pytorch를 이용한 MLP을 
<br> 추가하여 완성해볼 예정이다

## 데이터셋
HOG를 위한 데이터셋은 Kaggle을 통해서 숫자를 손을 통해서 표현한 이미지를 사용한다

사진

<br>예시로 하기 위해서 3을 가리키는 데이터를 사용할 예정이다

## 이미지 resize
먼저 주어진 이미지를 한번에 처리할 조각으로 나누어야 한다
<br>(8,8)의 크기로 진행을 하였다

## Calculate the gradient 
이제 gradient를 계산해야되는데
<br>먼저 horizontal 과 vertical gradient을 각각 측정한다

<br>gradient 측정을 위한 필터로 (-1,0,1) 필터를 사용한다
<br>다른 필터를 사용하여도 결과에는 큰 차이가 없다는 논문 결과가 있어서 추후에 실험 해볼 예정이다

사진

<br>필터를 적용하는 코드는 다음과 같다

사진

이중 for문을 통해서 해당 이미지에서 아까 resize하려고 하였던
<br>(8,8)크기가 bat_x, bat_y에 해당된다
<br>self.fil_sze는 필터의 사이즈 즉 3을 뜻한다
<br>self.pad와 self.stride는 padding과 stride의 개수를 의미하는데
<br>다음에 자세히 설명할 예정이라 일단은 넘어가겠다
<br>코드를 보시면 numpy의 slicing을 이용하여 필터의 사이즈만큼 3칸을
<br>필터와 곱을 하여 np.sum으로 합하는 것을 볼 수 있다

<br>예를 들어 1열이 [1,2,3,4,5,6,7,8]이라고 가정한다면
<br>필터가 [-1,0,1]이므로
<br>1번째로 1*-1 + 2*0 + 3*1으로 2가 나오게 된다
<br>이와 같이 이미지의 모든 좌표에 대하여 적용하게 된다

<br>첫번째 필터로는 X축의 gradient를 측정하고 두번째 필터로는 y축의 gradient를 측정하게 된다

