---
layout: post
title: (번역) Neural networks and Deep learning - Ch1. 뉴럴네트워크로 손글씨 숫지를 인식하기 - 1부 
category: neural networks and deep learning
tags: [neural network, perceptron, deep learning, 번역, 1장]
---
-**원저자: [Michael Neilson](http://michaelnielsen.org/)**<br>
-**원문주소: [http://neuralnetworksanddeeplearning.com/chap1.html](http://neuralnetworksanddeeplearning.com/chap1.html)**<br>
-**역자: [galji(지중현)](joonghyunji@gmail.com)**<br>
***본 번역의 무단 전재 및 재배포를 금지합니다.***
<br>
<br>

## 뉴럴네트워크로 손글씨 숫자를 인식하기 ## 

인간의 시각시스템은 이 세상에서 경이로운 것 중 하나이다. 아래의 필기체 숫자를 잠시 보자.

![handwritten digits](/deeplearning/assets/images/digits.png){: .center-image}

사람들은 이 숫자를 별 노력 없이 504192라고 인식할 것이다. 그런데 그 원리는 사실 복잡하다. 인간의 좌뇌와 우뇌에는 각각 1.4억 개의 뉴런과 그 뉴런들이 수백억 개나 연결된 V1이라고 불리는 ‘1차시각피질(primary visual cortex)’이 자리하고 있다. 점진적으로 고도의 영상 처리를 가능하게 하려면 V2, V3, V4 그리고 V5 같은 시각피질들도 시각시스템에 모두 연루되어야만 한다. 인간의 머릿속에는 수 억년 동안 진화하며 시각 세계를 이해하기 위해 훌륭하게 적응된 슈퍼컴퓨터가 있다. 필기체 숫자를 인식하기는 쉽지 않은데도 우리 인간은 경탄스럽게도 눈이 보는 것을 너무 잘 이해하는 것이다. 이런 과정은 무의식적으로 일어나기 때문에 시각시스템이 얼마나 힘든 문제를 해결하는지 그  진가를 알지 못한다.

위와 같은 숫자를 컴퓨터 프로그램을 작성해서 인식하도록 하면 시각패턴인식이 얼마나 어려운지 명백해진다. 아까는 쉬웠던 일이 갑자기 극도로 어려워지는 것이다. 우리가 어떻게 모양을 인지하는지 간단한 예를 들어보자. “9는 윗쪽에 고리 모양이 있고 오른쪽 아래에는 수직의 획이 있다.” 이것을 알고리즘으로 나타내는 것은 간단하지 않다. 이런 규칙들을 정밀하게 만들려고 시도하면 할수록 예외, 경고, 특별한 사례들로 수렁에 빠질 것이다. 절망적인 순간이다.

뉴럴네트워크는 이 문제를 다른 방식으로 접근한다. 발상은 다음과 같이

![mnist_100_digits](/deeplearning/assets/images/mnist_100_digits.png){: .center-image}

 훈련 사례(training example)라고 부를 수 있는 많은 손글씨 숫자들을 취합하는 것이다. 그다음에 훈련사례들을 가지고 학습할 수 있는 시스템을 만든다. 다시 말하자면, 뉴럴네트워크가 훈련사례들을 이용해서 손글씨 숫자를 인식하는 규칙들을 자동으로 추론하는 것이다. 훈련사례의 개수를 늘리면 뉴럴네트워크는 좀 더 많은 손글씨를 학습할 수 있으므로 결국 추론의 정확도는 올라간다. 여기에선 단지 100개의 훈련사례만 사용했지만 수천, 수백만 개 더 나아가 수십억 개의 훈련사례를 사용하면 손글씨 인식기의 성능을 훨씬 높일 수 있을 것이다.   

이번 장에서 우리는 뉴럴네트워크가 손글씨 숫자를 인식하도록 학습하는 컴퓨터 프로그램을 짤 것이다. 이건 코드가 74줄밖에 안 되고 특별한 뉴럴네트워크 라이브러리도 필요없다. 하지만 이 짧은 프로그램은 사람의 개입 없이 96%의 정확도로 숫자들을 인식할 수 있다. 다음 장들에서는 이 프로그램의 정확도를 점차 99%까지 향상할 것이다. 현재 가장 좋은 유료의 뉴럴네트워크 프로그램들은 카메라 이미지에서 은행의 수표나 편지의 주소를 인식하는 데 쓰일 정도로 성능이 좋다는 것을 알아두자.

당분간 손글씨 인식에 집중하기로 하자. 이것이 뉴럴네트워크를 공부하기 위한 일반적으로 훌륭한 전형적 문제이기 때문이다. 손글씨 인식은 도전적이지만 또한 균형이 잡힌 사례이다. 손글씨 숫자인식은 간단한 것이 아니지만, 해결책이 극도로 복잡하거나 엄청난 계산량이 필요한 만큼 어렵지는 않다. 게다가 딥러닝 같은 고급 기법을 이해하는데 훌륭한 방법이기도 하다. 따라서 이 책에서 반복적으로 손글씨 인식문제를 상기시킬 것이다. 이어 책 후반부에서는 컴퓨터비젼, 음성인식, 자연어 처리 등의 분야에서 이러한 아이디어가 어떻게 적용되는지 논의할 것이다.

컴퓨터 프로그램이 손글씨 숫자를 인식하게 하는 것만이 목표라면, 이번 장은 당연히 훨씬 짧았을 것이다. 하지만 우리는 뉴럴네트워크의 핵심인 퍼셉트론(perceptron)과 시그모이드 뉴런(sigmoid neuron) 같은 두 가지 인공뉴런과 확률적 경사하강법(stochastic gradient descent) 같은 표준학습 알고리즘을 섭렵해야 한다. 나는 핵심 아이디어들을 왜 그렇게 배치했는지 설명하고 뉴럴네트워크에 대한 독자의 직관을 세우는 데 집중할 것이다. 따라서 뉴럴네트워크에서 어떤 일들이 벌어지는지 기본적인 역학을 설명하는 것보다 더 긴 논의가 필요했다. 더 깊은 이해를 위해서는 이것이 바람직하다. 1장을 다 읽고 우리는 딥러닝이 무엇이며 왜 중요한지 이해하게 될 것이다. 

