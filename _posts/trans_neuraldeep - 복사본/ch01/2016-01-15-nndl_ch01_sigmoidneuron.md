---
layout: post
title: (번역) Neural networks and Deep learning - Ch1. 뉴럴네트워크로 손글씨 숫지를 인식하기 - 3부 
tags: [neural network, sigmoid neuron, deep learning, 번역, 1장]
---
-**원저자: [Michael Neilson](http://michaelnielsen.org/)**<br>
-**원문주소: [http://neural네트워크sanddeeplearning.com/chap1.html](http://neural네트워크sanddeeplearning.com/chap1.html)**<br>
-**역자: [galji(지중현)](joonghyunji@gmail.com)**<br>
***본 번역의 무단 전재 및 재배포를 금지합니다.***
<br>
<br>

###시그모이드 뉴런###

학습 알고리즘이라고 하니 좀 대단해 보인다. 하지만, 어떻게 뉴럴네트워크에다 이런 알고리즘을 적용할 수 있을까? 우리가 퍼셉트론을 어떤 문제를 해결하기 위해 사용한다고 가정해 보자. 예를들어, 손글씨 숫자를 스캔하여 픽셀 데이터를 얻고 그것을 네트워크의 입력으로 넣었다고 말이다. 그렇다면, 네트워크가 가중치와 편향치를 학습하여 스캔된 숫자를 정확히 분류하길 원할 것이다. 학습이 어떻게 작동하는지 보기 위해, 네트워크에서 가중치나 편향치를 살짝 변화시켜 보자. 가중치를 살짝 변경해서 네트워크에서 출력도 그에 맞게 적당히 변화하길 바라는 것이다. 금방 알게 되겠지만, 이런 속성이 학습을 가능하게 한다. 이것이 개략적으로 우리가 원하는 것이다 (당연히, 이 네트워크를 손글씨 인식에 쓰기엔 너무 단순하다).
 
 ![handwritten digits](/deeplearning/assets/images/tikz8.png){: .center-image}

만약 가중치나 편향치를 살짝 변화시킬때 출력도 조금만 변화될 뿐이라면, 이 사실로 볼때 좀 더 우리가 원하는 방향으로 네트워크가 작동하도록 가중치나 편향치를 변경시킬 수도 있을 이다. 예를 들어, 네트워크가 이미지를 숫자 "$9$"를 "$8$"이라고 잘못 분류했다고 가정하자. 이 때 이미지를 분류하여 숫자 "$9$"에 조금씩 가까워지도록 하기 위해  어떻게 가중치와 편향치를 변경해야 하는지가 문제이다. 조금씩 가중치와 편향치를 변화하여 반복하게 되면 점점 더 좋은 결과를 얻게된다. 이것이 네트워크가 학습하는 원리이다.


그런데 네트워크가 퍼셉트론으로 이루어져 있다면 위의 학습이 불가능하다. 당연하게도, 가중치와 편향치를 조금 건드리면 네트워크의 결과가 급격히 변동한다. 사실, $0$에서 $1$처럼 급격하게 바뀐다. 이러한 급격한 변동은 네트워크의 나머지 부분도 완전히 이해하기 어렵고 복잡하게 변하게 한다. 그래서 이미지가 숫자 "9"로 올바로 분류가 되는 한편, 다른 숫자 이미지를 인지하는 네트워크의 작용은 통제하기 어렵게 변할지도 모른다. 이 때문에,  네트워크가 가중치와 편향치를 점진적으로 변화시키는 방법이 좋은 결과로 수렴하기 어렵게 한다. 다행이 이러한 문제를 성공적으로 해결할 방법이 있긴 하다. 하지만, 퍼셉트론으로 이것을 어떻게 해결할 지 지금 당장은 불명확해 보인다.


이 문제는 새로운 인공뉴런의 종류인 시그모이드 뉴런(sigmoid neuron)으로 해결할 수 있다. 시그모이드 뉴런은 퍼셉트론과 비슷하지만, 가중치와 편향치를 약간 변동시키면 출력에도 약간의 변화만을 일으키도록 개선되었다. 이 사실이 퍼셉트론에 비해 시그모이드 뉴런의 네트워크가 더 잘 학습하는 중대한 요인이다.


좋다, 그럼 시그모이드 뉴런을 묘사해보자. 시그모이드 뉴런도 퍼셉트론을 그렸던 방식처럼 그려보자.

 ![handwritten digits](/deeplearning/assets/images/tikz9.png){: .center-image}
 
시그모이드 뉴런도 퍼셉트론처럼 $x_1$, $x_2,\ldots$같은 입력을 받는다. 그렇지만 $0$과 $1$ 대신에 그 사이값의 입력도 가능하다. 그래서 $0.638\ldots$ 같은 값도 시그모이드 뉴런에서는 유효한 입력이다. 또한, 시그모이드 뉴런에는 $w_1, w_2, \ldots$와 $b$ 같은 가중치와 총 편향치가 존재한다. 하지만 출력은 $0$이나 $1$ 아니다. 츨력은 실수로서 $\sigma(w \cdot x+b)$의 값을 가지는데, 여기서 $\sigma$는 *시그모이드 함수*이다. 시그모이드 함수의 정의는 다음과 같다:

\begin{eqnarray}
\label{eq:sig_func}
 \sigma(z) \equiv \frac{1}{1+e^{-z}}. \tag{3}\end{eqnarray} 

위 수식을 명시적으로 보이자면, 입력 $x_1, x_2,\ldots$, 가중치 $w_1, w_2,\ldots$ 그리고 총 편향치 $b$를 가진 시그모이드 뉴런은 아래와 같다.

\begin{eqnarray}
\label{eq:sig_func_2}
 \frac{1}{1+\exp(-\sum_j w_j x_j-b)}. \tag{4}\end{eqnarray}

처음 시그모이드 뉴런의 수식을 보면 퍼셉트론과는 매우 달라보인다. 시그모이드 함수의 대수적(algebraic) 형태는 이것에 익숙한 사람이 아니라면 뭔가 불투명하고 어지럽게 보일지 모른다. 당신이 이미 친숙한 경우가 아니라면 접근하기 어려워 보일지도 모른다. 복잡한 수식이 이해를 막는 장벽이 될 수는 없다. 왜냐하면 두 자기 형태의 뉴런 사이에는 많은 공통점이 있고, 시그모이드 함수의 대수적 형태가 퍼셉트론보다 좀 더 많은 기술적인 세부사항이 있는 것 뿐이다. 

퍼셉트론과의 유사성을 이해하기 위해, $z \equiv w \cdot x + b$으로 표시되는 $z$가 큰 양수라고 가정해보자. 그렇다면 $e^{-z}\approx 0$가 되며 $\sigma(z)\approx 1$이 된다. 다시 말해, $z$가 큰 양수라면, 시그모이드 뉴런의 출력은 대략 $1$이므로 퍼셉트론의 출력인 $1$과 매우 비슷해진다. 반대로, $z$값이 큰 음수라면 어떨까. 그렇다면 $e^{-z}\approx \infty$가 되며 $\sigma(z)\approx 0$이 된다. 그래서 만약 $z$가 큰 음수라면, 시그모이드 뉴런의 출력은 대략 $0$이므로 퍼셉트론의 출력인 $0$과 매우 비슷해진다. 퍼셉트론 모델과 큰 편차가 생기는 지점은 오로지 $w \cdot x+b$가 적당한 값을 가지는 구간 뿐이다. 


그렇다면 $\sigma$의 대수적 형태는 어떠한가? 그것은 어떻게 이해할 수 있을까? 사실,  $\sigma$의 정확한 형태보다 정말 중요한 것은 함수의 플롯(plot) 형태이다. 함수의 플롯은 함수의 좌표를 찍었을 때 그려지는 모양이다. 여기 아래 그림은 함수를 플롯한 것이다:


<div id="sigmoid_graph"><a name="sigmoid_graph"></a></div>
<script src="http://d3js.org/d3.v3.min.js"></script>
<script>
function s(x) {return 1/(1+Math.exp(-x));}
var m = [40, 120, 50, 120];
var height = 290 - m[0] - m[2];
var width = 600 - m[1] - m[3];
var xmin = -5;
var xmax = 5;
var sample = 400;
var x1 = d3.scale.linear().domain([0, sample]).range([xmin, xmax]);
var data = d3.range(sample).map(function(d){ return {
        x: x1(d), 
        y: s(x1(d))}; 
    });
var x = d3.scale.linear().domain([xmin, xmax]).range([0, width]);
var y = d3.scale.linear()
                .domain([0, 1])
                .range([height, 0]);
var line = d3.svg.line()
    .x(function(d) { return x(d.x); })
    .y(function(d) { return y(d.y); })
var graph = d3.select("#sigmoid_graph")
    .append("svg")
    .attr("width", width + m[1] + m[3])
    .attr("height", height + m[0] + m[2])
    .append("g")
    .attr("transform", "translate(" + m[3] + "," + m[0] + ")");
var xAxis = d3.svg.axis()
                  .scale(x)
                  .tickValues(d3.range(-4, 5, 1))
                  .orient("bottom")
graph.append("g")
    .attr("class", "x axis")
    .attr("transform", "translate(0, " + height + ")")
    .call(xAxis);
var yAxis = d3.svg.axis()
                  .scale(y)
                  .tickValues(d3.range(0, 1.01, 0.2))
                  .orient("left")
                  .ticks(5)
graph.append("g")
    .attr("class", "y axis")
    .call(yAxis);
graph.append("path").attr("d", line(data));
graph.append("text")
     .attr("class", "x label")
     .attr("text-anchor", "end")
     .attr("x", width/2)
     .attr("y", height+35)
     .text("z");
graph.append("text")
        .attr("x", (width / 2))             
        .attr("y", -10)
        .attr("text-anchor", "middle")  
        .style("font-size", "16px") 
        .text("시그모이드 함수");
</script>

<p>이 형태는 스텝함수(step function)처럼 원본 함수를 평평하게 만든 것이다: 
</p>
<div id="step_graph"></div>
<script>
function s(x) {return x < 0 ? 0 : 1;}
var m = [40, 120, 50, 120];
var height = 290 - m[0] - m[2];
var width = 600 - m[1] - m[3];
var xmin = -5;
var xmax = 5;
var sample = 400;
var x1 = d3.scale.linear().domain([0, sample]).range([xmin, xmax]);
var data = d3.range(sample).map(function(d){ return {
        x: x1(d), 
        y: s(x1(d))}; 
    });
var x = d3.scale.linear().domain([xmin, xmax]).range([0, width]);
var y = d3.scale.linear()
                .domain([0,1])
                .range([height, 0]);
var line = d3.svg.line()
    .x(function(d) { return x(d.x); })
    .y(function(d) { return y(d.y); })
var graph = d3.select("#step_graph")
    .append("svg")
    .attr("width", width + m[1] + m[3])
    .attr("height", height + m[0] + m[2])
    .append("g")
    .attr("transform", "translate(" + m[3] + "," + m[0] + ")");
var xAxis = d3.svg.axis()
                  .scale(x)
                  .tickValues(d3.range(-4, 5, 1))
                  .orient("bottom")
graph.append("g")
    .attr("class", "x axis")
    .attr("transform", "translate(0, " + height + ")")
    .call(xAxis);
var yAxis = d3.svg.axis()
                  .scale(y)
                  .tickValues(d3.range(0, 1.01, 0.2))
                  .orient("left")
                  .ticks(5)
graph.append("g")
    .attr("class", "y axis")
    .call(yAxis);
graph.append("path").attr("d", line(data));
graph.append("text")
     .attr("class", "x label")
     .attr("text-anchor", "end")
     .attr("x", width/2)
     .attr("y", height+35)
     .text("z");
graph.append("text")
        .attr("x", (width / 2))             
        .attr("y", -10)
        .attr("text-anchor", "middle")  
        .style("font-size", "16px") 
        .text("스텝함수");
</script>


만약 $\sigma$가 스텝함수였다면, 시그모이드 뉴런은 퍼셉트론과 같다. 왜냐하면, 시그모이드 뉴런의 출력을 1이나 0으로 결정하는 요인은 $w \cdot x + b$값이 양수인지 음수인지에만 관련있기 때문이다. 하지만 원본 $\sigma$ 함수를 사용하면서 퍼셉트론을 매끈한 모양으로 만들 수 있다. 따라서 세부적인 형태보다는 이러한 매끈함이 $\sigma$ 함수의 핵심이다. $\sigma$의 매끈함은 작은 가중치 변화량 $\Delta w_j$와 작은 편향치 변화량 $\Delta b$이 뉴런에서 작은 출력 변화 $\Delta \mbox{output}$를 줄 수 있다는 것을 의미한다. 미분을 이용하면 $\Delta \mbox{output}$이 대략 아래와 같이 근사된다.

\begin{eqnarray} 
\label{eq:delta_output}
  \Delta \mbox{output} \approx \sum_j \left(\frac{\partial \, \mbox{output}}{\partial w_j}
  \Delta w_j + \frac{\partial \, \mbox{output}}{\partial b} \Delta b\right).
\tag{5}\end{eqnarray}



 여러분이 편도함수(partial derivatives)에 익숙하지 않다고 해서 두려워 하지 마시라!  위 수식은 편미분이 등장해 복잡해보이지만, 사실 매우 간단한 걸 이야기해 준다 (좋은 소식이다): $\Delta \mbox{output}$는 $\Delta w_j$와 $\Delta b$의 선형 함수ㅔ 불과하다. 이러한 선형성은 가중치와 편향치를 살짝 변화시켜서 우리가 희망하는 어떤 작은 변화를 출력에서 얻기 쉽다는 것을 의미한다. 따라서, 시그모이드 뉴런이 퍼셉트론의 형태와 질적으로 많이 비슷해보일지라도, 원하는 출력을 얻기 위해 가중치와 편향치를 바꾸는 방법을 알아내기 훨씬 쉽다.

$\sigma$ 함수 본연의 형식이 아니라 플롯 모양이 훨씬 중요하다면, 왜 우리는 수식 \eqref{eq:sig_func}은 어떻게 나오게 된걸까? 실제로, 우리는 이따금씩 몇몇 다른 활성화 함수(activation function)들인 $f(\cdot)$에 대하여 출력이 $f(w \cdot x + b)$인 뉴런을 다룰 것이다다. 다른 활성화 함수를 사용할때 가장 큰 변화는 수식 \eqref{eq:delta_output}에 있는 특정한 변수에 대한 편미분값의 변화이다. 나중에 알게되겠지만, $\sigma$함수를 이용해서 이러한 편도함수를 계산하는 것은 지수함수의 미분에서 특별한 성질을 이용하면 간단하다. 어쨌든, $\sigma$함수는 뉴럴네트워크에서 보편적으로 쓰이며, 이 책에서 가장 많이 등장하는 활성화 함수다.

시그모이드 뉴런의 출력을 어떻게 해석해야 할까? 퍼셉트론과 시그모이드 뉴런의 가장 큰 차이는 시그모이드 뉴런의 출력이 $0$과 $1$만이 아니라는 것이다. 출력값이 $0$과 $1$ 사이의 실수이므로 $0.173$이나 $0.689$같은 값들도 타당한 출력이다. 이러한 속성은 매우 유용하다. 예를 들어, 한 이미지에 있는 픽셀값들의 평균 밝기(intensity)를 뉴럴네트워크의 출력으로 나타낼 수 있게 된다. 하지만 이러한 속성도 가끔은 골칫거리가 될 수 있다. 만약 "입력 이미지가 $9$다"나 "입력 이미지가 $9$가 아니다"처럼 두 가지 상태만을 출력으로 하는 네트워크가 있다고 생각해 보자. 당연히, 퍼셉트론처럼 0과 1읠 출력을 준다면 문제는 쉽다. 하지만, 실제로 중간에 있는 출력에 대해 분류할 수 있도록 하나의 규칙처럼 쓰이는 관례(convention)을 정할 수 있다. 예를 들어, $0.5$와 같거나 더 큰 출력이라면 "$9$"로 분류 하고, 작다면 "$9$"가 아닌 것으로 분류하는 것이다. 혼란을 피하기 위해서, 매번 그것이 관례임을 분명하게 선언하고 사용할 것임을 알려드린다. 


###연습문제###

- <p style="font-weight: bold"> 퍼셉트론을 모방하는 시그모이드 뉴런, 파트 1</p>
 -- 퍼셉트론 네트워크의 모든 가중치와 편향치에다 양의 상수 $c$를 곱하더라도 네트워크의 거동이 변하지 않음을 보여라.
<br>
- <p style="font-weight: bold"> 퍼셉트론을 모방하는 시그모이드 뉴런, 파트 2</p>
 -- 위의 문제와 설정이 같다고 전제한다.  퍼셉트론 네트워크에 모든 입력이 정해졌다고 가정하자. 실제 입력값은 필요가 없고 그저 고정된 입력이 필요할 뿐이다. 또한, 네트워크에서 어떤 특정한 퍼셉트론으로의 입력 $x$가 $w\cdot x + b \neq 0$을 만족하는 가중치와 편향치들이 있다고 가정하자. 이제 네트워크에서 모든 퍼셉트론 시그모이드 뉴런으로 교체하자. 그리고 모든 가중치와 편향치에다 양의 상수 $c$를를 곱하자. $c\to\infty$라면, 이러한 시그모이드 뉴런 네트워크의 거동은 퍼셉트론 네트워크와 정확히 일치함을 보여라. 퍼셉트론 중 하나가 $w\cdot x + b = 0$인 경우 어떻게 이것이 실패하는지 보여라.
