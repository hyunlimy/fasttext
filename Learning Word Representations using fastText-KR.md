
# Learning word representations using fastText

***
본 튜토리얼은 [다음의 튜토리얼](https://github.com/facebookresearch/fastText/blob/master/tutorials/unsupervised-learning.md)을 번역한 것입니다. 번역 오류나 오탈자 등에 대하여서 자유롭게 피드백 주시면 감사하겠습니다.<br>
*주: 변역자의 주석에 대해 이와 같이 이태릭체로 기술하였습니다.*
***

최근의 머신러닝에서 널리 사용되는 아이디어는 단어를 벡터로 표현하는 것입니다. 이러한 벡터는 단어 유추(analogies) 또는 의미(semantic)와 같은 언어에 숨겨진 정보를 캡쳐합니다. 또한 텍스트 분류기의 성능을 향상시키는데도 사용됩니다.

이 튜토리얼에서는 fastText라는 도구를 활용하여 단어 벡터를 생성하는 방법을 다룹니다. fastText를 다운로드하고 설치하는 단계는 [텍스트 분류에 대한 튜토리얼](https://github.com/facebookresearch/fastText/blob/master/tutorials/supervised-learning.md) 첫 단계를 참고하십시오.<br>

## Getting the data
***
Word vector를 계산하려면 대규모의 text corpus가 필요합니다. corpus에 따라 word vectors에서 포착되는 정보가 달라집니다.<br>
이 튜토리얼에서는 Wikipedia의 article을 활용하지만, 뉴스나 웹 크롤링과 같은 다른 소스도 사용될 수 있습니다. *([여기서](http://statmt.org)  더 많은 예시를 확인할 수 있습니다.)*<br>
아래의 코드를 통해 Wikipedia의 raw dump를 다운로드 할 수 있습니다.


*주: 데이터 용량이 13GB 입니다. 다운로드에 시간이 걸릴 수 있습니다.*

```shell
$ wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
```

튜토리얼을 위해 Wikipedia data의 처음 1 Billion Byte만 가져와서 실행합니다. [다음](http://mattmahoney.net)의 사이트에서 찾을 수 있습니다.<br>
다운로드 하는 코드는 아래와 같습니다.

*주: 여기서는 data폴더를 생성하여 다운로드 하였습니다. 데이터 용량은 380MB 입니다.* 

```shell
$ mkdir data
$ wget -c http://mattmahoney.net/dc/enwik9.zip -P data
$ unzip data/enwik9.zip -d data
```

Wikipedia raw dump에는 많은 HTML/XML 데이터가 포함되어 있습니다. 우리는 fastText에서 번들로 제공하는 wikifil.pl 스크립트를 이용하여 pre-processing 합니다.<br>
(이 스크립트는 Matt Mahoney가 개발했으며, 그의 웹사이트에서 찾을 수 있습니다.)

*주: wikifil.pl 파일은 fastText를 git을 통해 다운로드 하셨으면 해당 폴더의 최상위단에 존재합니다.*

```shell
$ perl wikifil.pl data/enwik9 > data/fil9
```

아래의 코드를 통해 파일을 확인할 수 있습니다.

*주: 본래 튜토리얼에는 'data/text9'를 열라고 되어있으나, 'data/fil9'로 스크립트를 실행하였기에 그에 맞추어 변경하였습니다*

```shell
$ head -c 80 data/fil9
anarchism originated as a term of abuse first used against early working class
```

Word vector를 학습하기 위한 데이터 pre-processing이 잘 수행된 것을 확인할 수 있습니다.

## Training word vectors
***
Word Vector에 대한 학습은 아래 한 줄로 명령으로 간단히 수행됩니다.

*주: 다운받은 fasttext 폴더 내의 fasttext파일을 실행시킵니다. 본 튜토리얼에서는 fasttext 폴더 내에서 input data와 output data를 모두 불러온다고 가정하였고, 경로가 다를 경우 환경에 맞게 조정이 필요합니다.*

**in case of skipgram**

```shell
$ mkdir result
$ ./fasttext skipgram -input data/fil9 -output result/fil9
```

**in case of cbow**
```shell
$ ./fasttext cbow -input data/fil9 -output result/fil9
```

이 command line의 의미는 다음과 같습니다:<br>
./fasttext 는 binary fastText 실행 파일을 호출하고 **skipgram** 모델을 사용하도록 합니다. (**cbow** 모델도 마찬가지 입니다.)<br>
그 다음 '***-input***' 옵션을 통해 pre-processing이 완료된 데이터를 집어 넣고, word representation이 저장될 위치에 대하여 '***-output***' 옵션을 통해 지정합니다.

fastText가 실행되는 동안 진행 상황과 완료 예상시간이 스크린에 표시됩니다. 프로그램이 종료되고 나면, 결과 디렉토리에 '***fil9.bin***' 파일과 '***fil9.vec***' 두개의 파일이 있어야 합니다.

```shell
$ ls -l result
-rw-rw-r-- 1 lims1 lims1 978480868  JUN 20 15:52 fil9.bin
-rw-rw-r-- 1 lims1 lims1 191094815  JUN 20 15:52 fil9.vec
```

__fil9.bin__ 파일은 전체 fastText 모델을 저장하고 연속적으로 로드할 수 있는 binary file 입니다.<br>
__fil9.vec__ 파일은 한 줄에 하나씩의 단어와 word vector가 포함되어 있는 텍스트 파일입니다.

첫 번째 줄의 헤더에는 단어 수와 벡터의 차원을 포함하고 있습니다.<br>
다음 줄부터 단어들과 각 단어들이 가지고 있는 word vector들이 나오고, 빈도의 내림차순으로 정렬되어 나타납니다.

```shell
$ head -n 4 result/fil9.vec
the -0.10363 -0.063669 0.032436 -0.040798 0.53749 0.00097867 0.10083 0.24829 ...
of -0.0083724 0.0059414 -0.046618 -0.072735 0.83007 0.038895 -0.13634 0.60063 ...
one 0.32731 0.044409 -0.46484 0.14716 0.7431 0.24684 -0.11301 0.51721 0.73262 ...
```


# skipgram versus cbow

***
fastTexts는 word representation을 계산하는 모델로서 두 가지를 제공합니다.<br>
**skipgram**과 **cbow**(continuous-bag-of-words) 입니다.

**skipgram** 모델은 주변에 등장하는 단어들을 바탕으로 목표 단어를 예측하는 방법을 학습합니다. 반면, **cbow** 모델은 문맥(context)에 따라 목표 단어를 예측하는 방법을 학습합니다. 문맥은 목표 단어 주변의 fixed size window에 포함되는 bag of the words 로 표현됩니다.

예시를 통해 확인해보면 다음과 같습니다.<br>

>'Poets have been mysteriously silent on the subject of cheese' <br>

라는 문장이 있을 때 목표 단어가 'silent'라 하면,<br>

**skipgram** 모델은 'subject\*'와 'mysterious\*\*' 같이 목표 단어 주변의 단어로부터 예측을 수행합니다.<br>
**cbow** 모델은 단어 주변의 window에 포함되는 모든 단어들{'been', 'mysterious', 'on', 'the'}의 벡터값들을 모두 더해 목표 단어를 예측합니다.

아래의 그림은 두 모델의 차이점을 다른 예를 통해 요약하여 설명하는 그림입니다.

<img src='cbo_vs_skipgram.png'/>

Practical하게, **skipgram**이 subword information에 대해 **cbow**보다 더 잘 동작한다고 알려져 있습니다.

## Playing with the parameters

지금까지는 fastText를 default parameter로 돌렸습니다. 그러나 일부 데이터에 따라 default가 optimal이 아닐 수 있습니다.<br>
아래의 튜토리얼에서는 word vector 생성을 위한 key parameter에 대하여 소개하겠습니다.

모델에서 가장 중요한 파라미터는 벡터 공간의 크기(dimension)와 subwords의 사이즈 범위(range of size for the subwords)입니다.<br>

**dimension**(***dim***)은 vector 공간의 크기를 조절하고, 이 값이 커질수록 더 많은 정보를 얻을 수 있지만 더 많은 데이터를 학습해야 합니다.<br>
또한 이 값이 너무 크면 훈련하기 어려워지고 느려집니다. 기본적으로 **dimension** 값으로 100을 활용하지만, 100~300 사이의 값이 일반적으로 많이 사용됩니다.<br>

**subword**는 단어 내에 minimun size(***nmin***)와 maximal size(***nmax***) 사이의 값을 가지는 모든 substring을 의미합니다.<br>
기본적으로 3~6자 사이의 모든 subword를 활용하지만, 다른 언어에서는 다른 범위가 더 적절할 수 있습니다.

이 변수들은 다음과 같이 사용할 수 있습니다.

```shell
$ ./fasttext skipgram -input data/fil9 -output result/fil9 -nmin 2 -nmax 5 -dim 300
```


보유하고 있는 데이터의 양에 따라, training 시 활용되는 parameter를 수정해야 할 수도 있습니다.<br>

**epoch** 는 데이터에서 몇 번의 loop가 돌게 할 것인지 결정합니다.<br>
default로, 5번이 지정되어 있습니다. 만약 데이터가 매우 크다면, 반복횟수를 줄이는 것이 좋습니다.

또 다른 주요한 parameter는 **learning rate** (***-lr***)입니다.<br>
학습률 값이 클수록 모델이 수렴하는 속도가 빨라지겠지만, overfitting이 발생할 위험이 있습니다. default 값은 0.05이며 좋은 절충안(compromise)입니다.<br>학습률을 변경하고 싶다면, [0.1, 1]의 범위를 유지하는 것을 추천합니다.

```shell
$ ./fasttext skipgram -input data/fil9 -output result/fil9 -epoch 1 -lr 0.5
```

마지막으로, fastText는 12개의 multi-thread를 기본으로 사용합니다. 만약 CPU core수가 적다면(만약 4개라면), ***-thread*** 옵션을 통해 쉽게 변경해 줄 수 있습니다.

```shell
$ ./fasttest skipgram -input data/fil9 -output result/fil9 -thread 4
```

# Printing word vectors
***

***fil9.vec*** 파일에서 직접 word vector를 검색하고 인쇄하는 작업은 꽤 번거로울 수 있습니다.<br>
fastText에서는 이를 위해 ***print-word-vectors*** 함수를 지원합니다.

예를 들어, "*asparagus*", "*pidgey*", "*yellow*" 와 같은 단어의 word vector를 다음과 같이 출력할 수 있습니다.<br>

*주: 본래의 튜토리얼에는 'print-vectors' 함수로 작성되어 있으나, 해당 함수가 존재하지 않고 'print-word-vectors'만 존재하기에 이에 맞추어 수정하였습니다.*

```shell
$ echo "asparagus pidgey yellow" | ./fasttext print-word-vectors result/fil9.bin
asparagus 0.46826 -0.20187 -0.29122 -0.17918 0.31289 -0.31679 0.17828 -0.04418 ...
pidgey -0.16065 -0.45867 0.10565 0.036952 -0.11482 0.030053 0.12115 0.39725 ...
yellow -0.39965 -0.41068 0.067086 -0.034611 0.15246 -0.12208 -0.040719 -0.30155 ...
```

좋은 특징 중 하나는 보유하고 있는 데이터에 등장하지 않은 데이터들에 대하여서도 query를 할 수 있습니다.<br>
실제로 단어는 substrings의 합들로 표현됩니다.<br>
알려지지 않은 단어가 보유하고 있는 데이터들의 substrings의 조합으로 표현 가능하다면, 값을 출력할 수 있습니다.

이 예시로써 오탈자가 존재하는 단어에 대해 query 해보겠습니다.

```shell
$ echo "enviroment" | ./fasttext print-word-vectors result/fil9.bin
enviroment 0.015178 -0.082941 -0.06693 0.17505 -0.090651 -0.22564 -0.24376 -0.05723 ...
```

오탈자가 존재하는 query에 대해서도 잘 처리하는 것을 확인할 수 있습니다. 어느 수준까지의 variation된 query를 처리할 수 있는지 다음 session에서 확인해 보겠습니다.

## Nearest neighbor queries
***

Word vector의 품질을 확인하는 방법 중 가장 간단한 방법은 nearest neighbor를 확인하는 것 입니다.<br>
이것은 vector가 담고 있는 의미 정보가 어떤 타입인지 파악할 수 있도록 해줍니다.

***nn*** 함수를 이용하여 이를 파악할 수 있습니다.<br>
예를 들어, 다음 명령을 수행하여 해당 단어와 가장 가까운 10개의 단어를 출력할 수 있습니다.

```shell
$ ./fasttext nn result/fil9.bin
Pre-computing word vectors... done.
```

이후, query 단어를 입력하라는 메시지가 표시됩니다. 여기선 asparagus를 입력해보겠습니다.

```
Query word? asparagus
asparagus 0.945553
paragus 0.83575
cucumis 0.736204
arviragus 0.736065
chickpea 0.735916
horseradish 0.732933
fagus 0.730422
beetroot 0.730263
officinale 0.728787
esculentum 0.728605
```


야채와 관련된 vector들이 비슷해 보입니다. 가장 가까운 단어는 asparagus 그 자체임을 볼 수 있습니다.

앞서 확인했던 오탈자에 대하여 query 해 보면 결과는 다음과 같습니다.

```
Query word?  enviroment
enviromental 0.87566
environ 0.831924
enviro 0.822801
environnement 0.778086
enviromission 0.762148
environs 0.756486
realclimate 0.727037
carfree 0.68649
ecomuseum 0.68377
biomedcentral 0.677437
```

단어에 포함된 정보 덕분에, 철자가 틀린 단어가 주어져도 잘 맞는 것을 확인할 수 있습니다. 완벽하진 않지만, 주요 정보가 잘 추출되었다고 볼 수 있습니다.

### Measure of similarity
***

nearest neighbor를 찾기 위해, 우리는 단어 간 similarity에 대한 score를 계산해야 합니다. 단어는 연속적은 word vector 값으로 저장되기 때문에 간단한 similarities를 적용하여 계산해 낼 수 있습니다. 특히 두 vector 사이의 cosine 값을 사용합니다. 이 similarity는 한 단어와 다른 모든 단어 사이에서 계산되며 가장 유사한 10개의 단어가 표시됩니다. 물론, 해당 단어와 동일한 단어가 있으면 가장 위에 표시되고, similarity가 1로 표현될 것입니다. 

## Word analogies
***

위와 비슷한 개념으로, word analogies를 계산할 수도 있습니다.<br>
아래의 예시에서, berlin-Germany 관계와 비슷한 '?'-France에 해당되는 단어를 우리 모델이 유추할 수 있는지 확인해 보겠습니다.

*주: Word analogies는 두 단어간의 관계를 의미합니다. 예를 들어, {부분:전체} analogy type의 예시는 {날개:비행기}가 있을 수 있고, {제품:용도} analogy type의 예시는 {연필:필기} 가 있을 수 있습니다. 자세한 사항은 [여기](http://www.literarydevices.com/analogy/)를 참고하실 수 있습니다.*

이것은 ***analogies*** 함수를 통해 수행됩니다. 단어 삼중항(word triplet)을 input으로 받고(Germany Berlin France), analogy를 계산해 출력합니다.

```shell
$ ./fasttext analogies result/fil9.bin
Pre-computing word vectors... done.
```

```shell
Query triplet (A - B + C)? berlin germany france
paris 0.893499
louveciennes 0.775256
montpellier 0.751376
bourges 0.751251
valenciennes 0.748544
avignon 0.745142
strasbourg 0.743762
pompignan 0.74314
dubourg 0.740164
bourget 0.738653
```

우리 모델이 *paris* 라는 결과값을 올바르게 반환하였습니다. 조금 더 불 명확한 사례에 대해서도 살펴보겠습니다.

```shell
Query triplet (A - B + C)?  psx sony nintendo
nintendogs 0.691441
gamecube 0.670561
kart 0.652065
snes 0.628259
gba 0.626141
arcade 0.618393
sega 0.612909
minigame 0.60529
troopa 0.604168
gameboy 0.602926
```

우리 모델이 *{psx-sony}* 에 대응하는 *nintendo*의 analogy가 *nintendogs* 및 *gamecube*로 올바르게 예측하였습니다.<br>
물론 analogy의 품질은 모델을 훈련하는데 사용된 데이터에 따라 달라지며 학습된 데이터 집합 내에서의 유추만 기대할 수 있습니다.

## Importance of character n-grams
***
subword-level의 정보를 활용하면 unknown 단어들에 대해 vector를 구축할 때 매우 유용합니다.<br>
예를 들어, *gearshift*라는 단어는 Wikipedia 데이터에 존재하지 않지만 query를 진행할 수 있습니다.

```shell
$ ./fasttext nn result/fil9.bin
Pre-computing word vectors... done.
Query word? gearshift
gearing 0.803886
flywheel 0.799993
flywheels 0.796023
gears 0.786624
daisywheel 0.772059
driveshafts 0.764756
driveshaft 0.758708
cogwheels 0.757592
crankshaft 0.749779
crankshafts 0.740112
```

검색된 단어들의 대부분은 문자열을 공유하는 것이 많지만, *'cogwheels'*와 같이 일부는 다른 경우도 있습니다. 

이제 subword 정보를 통해 unknown 단어를 검색할 수 있음을 확인하였으니, subword 정보를 사용하지 않은 모델과 비교하여 봅시다.<br>
subword 정보 없이 모델을 학습시키려면 아래와 같이 수행할 수 있습니다.

```shell
$ ./fasttext skipgram -input data/fil9 -output result/fil9-none -maxn 0
```

학습에 대한 결과는 ***result/fil9-none.bin*** 과 ***result/fil9-none.vec*** 으로 저장될 것입니다.

차이점을 설명하기 위해 "*accommodation*" 의 오탈자 "*accomodation*" 과 같이 Wikipedia에서 거의 등장하지 않는 단어를 확인해 보겠습니다.<br>
subword 정보를 사용하지 않고 얻은 nearest neighbor의 결과는 다음과 같습니다.

```shell
$ ./fasttext nn result/fil9-none.bin
Pre-computing word vectors... done.
Query word? accomodation
hostelling 0.787859
administrational 0.782827
accomodations 0.782042
turism 0.766875
dachas 0.757473
directgov 0.752838
cbds 0.742739
sunnhordland 0.740666
sverok 0.740221
uutela 0.738532
```

결과도 크게 의미가 없고, 도출되는 단어들이 의미가 연관되지 않는 것들이 많습니다.<br>
반면에, subword 정보를 사용하도록 한 모델에 대한 동일한 query 결과는 다음과 같습니다.

```shell
$ ./fasttext nn result/fil9.bin
Pre-computing word vectors... done.
Query word? acomodation
accomodation 0.938892
accomodations 0.89894
accommodation 0.864848
accommodations 0.834365
accommodative 0.771078
amenities 0.700895
lodging 0.687514
catering 0.681449
campgrounds 0.681039
amenity 0.673948
```

nearest neighbor로 도출되는 단어들이 "*accomodation*" 이라는 오탈자의 변형들 입니다. 우리는 이를 통해 "*amenties*" 나 "*lodging*" 과 같이 의미적으로 비슷한 단어들도 등장함을 확인할 수 있습니다.

## Conclusion
***
이 튜토리얼에서 우리는 Wikipedia에서 어떻게 word vector를 추출할 수 있는지 살펴보았습니다. 이러한 방법으로 모든 언어들에 대해서 모델을 학습할 수 있으며, default setting으로 pre-training 된 294개의 모델을 [여기](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md)서 다운받으실 수 있습니다.

## [Appendix A] Using wikipdeia raw dump (EN)
***
***주: Appendix 의 모든 내용은 번역자가 별도 추가한 것입니다.***

본 튜토리얼의 초반에 다운받았던 wikipedia dump 전체 데이터를 활용해 skipgram 모델을 생성 해보겠습니다.<br>
데이터의 용량이 13GB로 매우 커서 작업하는 환경에 따라 매우 오랜 시간이 걸릴 수 있습니다.

bz2 파일을 풀기 위해서 ***bzip2*** 명령어를 사용합니다. <br>
본 튜토리얼에서는 압축해제 후 본래 archive를 유지하기 위해 ***-dk*** 옵션을 활용합니다.<br>
원본 archive를 삭제하시려면 ***-d*** 옵션을 사용하지면 됩니다.

```shell
$ bzip2 -dk enwiki-latest-pages-articles.xml.bz2
```


압축이 다 풀렸으면, fastText에서 번들로 제공하는 wikifil.pl 스크립트를 활용하여 pre-processing을 수행합니다.<br>
*(참고) 20-cores Xeon 2.5GHz, Ubuntu 14.04 환경에서 약 1~2시간 소요되었습니다.*

```shell
$ perl wikifil.pl data/enwiki-latest-pages-articles.xml > data/enwiki
```

아래의 코드를 통해 pre-processing된 파일을 확인할 수 있습니다.

```shell
$ head -c 80 data/enwiki
anarchism originated as a term of abuse first used against early working class
```

skipgram 모델의 학습을 수행합니다.<br>
*(참고) 20-cores Xeon 2.5GHz, Ubuntu 14.04 환경에서 약 7~8시간 소요되었습니다.*

```shell
$ ./fasttext skipgram -input data/enwiki -output result/enwiki
```

#### Printing word vectors
***
학습된 모델에, 상기 튜토리얼과 동일한 명령어들을 수행하여 데이터가 커졌을때 얼마나 더 성능이 좋아지는지 확인 해보겠습니다.

```shell
$ echo "asparagus pidgey yellow" | ./fasttext print-word-vectors result/enwiki.bin
asparagus -0.39733 0.26447 -0.55391 -0.025635 -0.65991 0.016171 -0.078411 0.031118 0.096442 ...
pidgey 0.14909 0.71044 -0.78849 0.21748 -0.52081 0.14747 -0.10854 0.28269 0.50547 0.35947 -0.017427 ...
yellow -0.27953 -0.081612 -0.53538 -0.12526 0.02019 0.15286 -0.17611 0.085327 0.023446 0.56017 ...

```

```shell
$ echo "enviroment" | ./fasttext print-word-vectors result/fil9.bin
enviroment -0.0025224 -0.016557 -0.3852 0.0687 -0.63787 -0.18916 -0.059942 0.0017906 -0.11485 0.50566 ...
```

#### Nearest neighbor
***

```shell
$ ./fasttext nn result/enwiki.bin
Pre-computing word vectors... done.

Query word? asparagus
sparagus 0.854393
toastedasparagus 0.839247
lettuce 0.832112
spinach 0.830396
cabbage 0.829901
chicory 0.827725
phaseolus 0.825148
lettuces 0.823741
protasparagus 0.821911
wintercress 0.821394

Query word? enviroment
enviroments 0.959155
enviromental 0.927047
enviromentally 0.894434
envirome 0.888848
enviromena 0.876878
environemnt 0.829639
environmnet 0.810482
enviromentalist 0.809986
envirom 0.808629
enviromentalism 0.791934
```

정확한 철자인 asparagus에 대해서는 좋은 결과를 보여주지만, 오탈자인 enviroment에 대하여서는 오히려 적은 데이터를 사용하였을 때 보다 오탈자를 교정해주지 못하고 그대로 출력해버리는 경향이 있습니다.

#### Word analogies
***


```shell
$ ./fasttext analogies result/enwiki.bin
Pre-computing word vectors... done.

Query triplet (A - B + C)? berlin germany france
paris 0.920255
ferrand 0.825052
marseille 0.823585
toulouse 0.817713
rouen 0.807836
montrouge 0.805249
marseilles 0.801083
strasbourg 0.80037
montpellier 0.796843
grenoble 0.791756
```

적은 데이터로 학습했을때 보다 *paris* 를 올바르게 추론하는 확률이 늘어났습니다.

```shell
Query triplet (A - B + C)? psx sony nintendo
gamepot 0.813452
wii 0.811759
nintendojo 0.811007
wiiplaystation 0.808049
gamepen 0.796823
wiiware 0.796684
gamecube 0.793289
nintendog 0.790586
gamespc 0.788907
xplaystation 0.78838
```

정답이든 아니든, 답을 추론할 때 근거가 되는 확률값들이 전체적으로 올라간 것을 확인할 수 있습니다.

## [Appendix B] Using pre-trained model (KR)
***
***주: Appendix의 내용은 번역자가 별도 추가한 것입니다.***<br>
Appendix B 는 [다음](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md)의 내용을 참고하여 작성되었습니다.


#### Pre-trained vector model
***
fastText에서는 Wikipedia를 활용해 학습한 296개 언어의 pre-trained word vector를 제공합니다.<br>
제공되는 모델들은 모두 300 dimension의 vector space를 갖도록 학습되었고, [skipgram](https://arxiv.org/abs/1607.04606) 모델로 default parameter들로 학습되었습니다.<br>
296개 언어 안에는 물론 한국어도 포함되어 있습니다.<br>

pre-trained word vector는 [여기](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md)서 다운받으실 수 있습니다.

Appendix B에서는 한국어 데이터를 활용하는 법을 소개하고, 간략히 성능을 평가 해보겠습니다.

#### Format
***
미리 학습된 word vector는 fastText의 기본 format인 **binary(.bin)**와 **text(.vec)** 두 가지 모두로 제공됩니다.
text format의 한 줄에는 단어와 각 단어가 가지는 embedding이 나열되어 있습니다. 각 embedding 값 하나는 space 하나를 의미합니다. 즉, 한 줄에는 300개의 embedding 값들이 있습니다.<br>
단어는 빈도의 내림차순으로 정렬되어 있습니다.

#### Download model
***
우선 한국어로 사전학습된 모델을 [다운로드](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.ko.zip) 합니다. 여기서는 ***(bin+text)***파일을 다운받아 사용하겠습니다.<br>
한국어 기준으로, 용량은 약 4.7GB 가량 입니다.

다운로드 받은 파일을 압축해제 합니다.

```shell
$ unzip data/wiki.ko.zip -d result
```

#### Test model
***
pre-train 된 모델을 바탕으로 테스트를 진행합니다.

우선, 아래와 같이 단어들의 word vector를 출력할 수 있습니다.

```shell
$ echo "생수 노트북 시계" | ./fasttext print-word-vectors result/wiki.ko.bin
생수 0.44658 -0.31546 0.038056 0.013477 -0.26236 -0.29344 -0.054675 -0.1649 0.36899 0.081029 ...
노트북 -0.0010818 -0.14623 -0.11676 -0.018334 -0.15783 0.018404 0.31924 0.13268 -0.051212 0.35432 ...
시계 0.18623 0.12693 -0.29382 0.42726 0.36109 -0.10816 0.28113 -0.082197 0.52519 0.56978 0.15964 ...

```

오탈자가 존재하는 query에 대해서도 결과를 반환합니다.

```shell
$ echo "아메맄카노" | ./fasttext print-word-vectors result/wiki.ko.bin
아메맄카노 -0.035199 0.36601 -0.52651 -0.070094 -0.17861 -0.13543 -0.028647 -0.19833 -0.070591 -0.2899 ...
```

특정 단어와 가까운 위치에 있는 단어들을 뽑아 보겠습니다.

```shell
$ ./fasttext nn result/wiki.ko.bin
Pre-computing word vectors... done.
Query word? 아메리카노
아메리카노 0.835343
아메리아 0.82663
아메리카즈 0.816599
아메리카누 0.811307
아메리카스 0.807077
아메리카레아 0.805204
아메리코 0.803199
아메리 0.793916
아메리카나 0.787029
아메리카담비 0.779375
```

외래에어 대해 생소한 단어들도 조합되어 출력하는 것을 확인할 수 있습니다.

```shell
Query word? 환경
환경의 0.781143
환경만 0.758366
환경과 0.741286
환경도 0.740041
환경세 0.734693
환경인 0.732118
환경관련 0.718287
환경을 0.716883
환경뿐만 0.714664
물환경 0.713252

Query word? 수박
수박과 0.738163
수박풀 0.696854
수박은 0.681046
수박바 0.678488
수박이 0.675679
수박의 0.658078
나물 0.625704
수박을 0.61817
호박 0.605062
메밀 0.603226
```

비슷한 단어는 뽑아내는 것으로 보이긴 하나, 이대로 사용하기엔 무리가 있어 보입니다.<br>
실제 사용하기 위해서는 한국어 전처리 과정이 별도로 수행되어야 할 것 같습니다.

Word analogy 를 한국어에 대해 수행 해보겠습니다.

```shell
$ ./fasttext analogies result/wiki.ko.bin
Pre-computing word vectors... done.
Query triplet (A - B + C)?  도쿄 일본 미국
웨스트뉴욕 0.668017
뉴욕주립 0.659137
마이애미가든스 0.65438
조지워싱턴 0.653351
노던일리노이 0.651568
워싱턴 0.650401
아이오와시티 0.646704
애틀랜틱시티 0.644708
미주리시티 0.642263
뉴욕시티 0.638848
```

```
Query triplet (A - B + C)?  자동차 기름 냉장고
기름도 0.689083
기름방울 0.678654
기름과 0.657222
기름에 0.657205
기름이나 0.656353
기름기 0.648209
엿기름 0.644983
냉장고의 0.64145
참기름 0.639422
냉장고에 0.635078

Query triplet (A - B + C)? 빛 어둠 선
선에 0.6182
선이 0.6095
선과 0.601383
선은 0.600824
선을 0.594466
선의 0.551565
분선 0.546474
선에서 0.523081
선에도 0.520983
선이나 0.511143
```

fastText로 한글을 학습시키기 위해서는 전처리가 매우 중요할 것으로 보입니다.

***
번역 및 수정 : Hyunlim YANG (lims1@dgist.ac.kr), InfoLab., DGIST / 2017.06.22
***
