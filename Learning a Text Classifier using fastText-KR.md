
# Learning a Text Classifier using fastText
***
본 튜토리얼은 [다음의 튜토리얼](https://github.com/facebookresearch/fastText/blob/master/tutorials/supervised-learning.md)을 번역한 것입니다. 번역 오류나 오탈자 등에 대하여서 자유롭게 피드백 주시면 감사하겠습니다.<br>
*주: 변역자의 주석에 대해 이와 같이 이태릭체로 기술하였습니다.*
***
텍스트 분류 문제는 스팸 탐지, 감성분석(sentiment analysis), 자동 응답(smart replies)와 같이 다양한 응용프로그램들의 핵심 문제 입니다.<br> 
이 튜토리얼에서는 fastText라는 도구를 활용해 텍스트 분류기를 작성하는 방법을 설명합니다.

## What is the classification?
***
텍스트 분류의 목표는 이메일, 개시물, 문자 메시지, 제품리뷰 등의 문서를 하나 또는 여러 범주로 할당하는 것입니다. 이러한 범주는 리뷰 점수, 스팸여부, 작성된 언어 등 다양할 수 있습니다. 최근에는 이러한 문서분류기를 만드는 주요 접근 방식으로 기존의 데이터로부터 분류 규칙을 학습하는 기계학습 방식을 채택하는 경우가 많습니다. 기계학습에서의 분류기준을 만들기 위해서는 학습 대상이 되는 문서와 그 문서가 해당되는 범주(태그, 혹은 레이블)로 구성된 데이터셋이 필요합니다.

예를 들어, 본 튜토리얼에서는 stackexchange의 요리와 관련된 질문들을 자동으로 냄비(pot), 그릇(bowl), 베이킹(baking) 등과 같은 여러 태그들 중 하나로 분류하는 학습기를 만들어 보겠습니다.

## Installing fastText
***
본 튜토리얼의 첫번쨰 단계는 fastText를 설치하고 build하는 것입니다. 설치를 위해 사전에 C++11 을 지원하는 C++ compiler가 필요합니다.

우선, fastText repository를 [clone](https://help.github.com/articles/cloning-a-repository/) 합니다.

```shell
$ git clone git@github.com/facebookresearch/fastText.git
```

설치한 fastText 폴더로 이동하여 build 합니다.

```
$ cd fastText && make
```

아무 추가 인자(argument) 없이 binary 파일을 실행하면 high level documentation이 출력되고, fastText에서 지원하는 함수들과 그 사용예가 기술되어 있습니다.

```shell
$ ./fasttext
usage: fasttext <command> <args>

The commands supported by fasttext are:

supervised     train a supervised classifier
test           evaluate a supervised classifier
predict        predict most likely labels
predict-prob   predict most likely labels with probabilities
skipgram       train a skipgram model
cbow           train a cbow model
print-vectors  print vectors given a trained model
```

이 튜토리얼에서는 test classfier 학습과 관련된 함수인 "*supervised*" 함수와 "*test*", 그리고 "*predict*"와 하위명령어들을 사용합니다.<br>
fastText의 다른 기능에 대한 소개는 [Learning Word Representations using fastText](https://github.com/facebookresearch/fastText/blob/master/tutorials/unsupervised-learning.md) 튜토리얼을 참고하십시오.

## Getting and preparing the data
***
서론에서 소개하였듯이, 지도학습 분류기(supervised classifier)를 학습하기 위해 우리는 label된 데이터가 필요합니다.<br>
이 튜토리얼에서는 Stackexchange에 올라온 요리와 관련된 질문들의 주제를 자동으로 분류하는 모델을 만들어 보고자 합니다.<br>
Stackexchange의 요리 섹션 있는 질문들과 연관된 태그를 다운로드 합니다.

```shell
$ wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/cooking.stackexchange.tar.gz && tar xvzf cooking.stackexchange.tar.gz
$ head cooking.stackexchange.txt
```

텍스트 파일의 각 행에는 label의 목록들이 나오며, 그 뒤에 관련된 문서들이 나옵니다.<br>
모든 label은 ***"__label__"*** 로 시작되며, 이를 통해 fastText에서 레이블과 단어를 구분하여 인식합니다.<br>
모델은 문서내에 주어진 단어들을 바탕으로 해당 label이 무엇일지를 예측하도록 학습됩니다.


분류기를 학습시키기 전에, 우리는 데이터를 train set과 validation set으로 분류해야 합니다.<br>
validation set은 우리의 classifier가 새로운 데이터에 대해 얼마나 잘 예측하는가를 평가하는데 사용될 것입니다.

```shell
$ wc cooking.stackexchange.txt
  15404  169582 1401900 cooking.stackexchange.txt
```

우리의 전체 데이터셋은 15404건의 example을 포함하고 있습니다. 데이터를 12404건의 training set과 3000건의 validation set으로 나눕니다.

```shell
$ head -n 12404 cooking.stackexchange.txt > cooking.train
$ tail -n 3000 cooking.stackexchange.txt > cooking.valid
```

## Our first classifier
***
이제 첫번째 분류기를 학습시킬 준비가 다 되었습니다.

*주: 본 튜토리얼에서는 fastText binary file과 stackexchange 데이터 파일이 동일한 폴더 내에 저장되어있다고 간주하고 분석합니다. 만약 환경이 다르다면 그에 맞추어 변경해 주어야 합니다.*

```shell
$ ./fasttext supervised -input cooking.train -output model_cooking
Read 0M words
Number of words:  14543
Number of labels: 735
Progress: 100.0%  words/sec/thread: 50082  lr: 0.000000  loss: 10.082111  eta: 0h0m
```

***-input*** 옵션은 학습대상이 되는 학습데이터의 위치를 나타내고 ***-output*** 옵션은 생성된 모델을 어디에 저장할 겻인지를 결정합니다.<br> 학습이 끝나고 나면 ***model_cooking.bin*** 파일이 생상되고, 여기에 학습된 모델이 저장되어 있습니다.

아래의 명령어를 통해 학습된 모델을 바로 테스트할 수 있습니다.

```shell
$ ./fasttext predict model_cooking.bin -
```

위와 같이 명령을 작성한 뒤, 문장을 입력해서 바로 테스트 해볼 수 있습니다.<br>
다음의 문장을 입력 해보겠습니다.
> Which baking dish is best to bake a banana bread?


```shell
$ ./fasttext predict model_booking.bin -
Which baking dish is best to bake a banana bread?
__label__baking
```

모델이 ***baking*** tag를 예측하였고, 질문과 잘 맞는 것 같습니다. <br>
마찬가지로 두번째 문장을 입력 해보겠습니다.
> Why not put knives in the dishwasher?

```shell
$ ./fasttext predict model_booking.bin -
Why not put knives in the dishwasher?
__label__food-safety
```

모델이 ***food-safety*** 라고 예측을 하였지만, 질문과 크게 관련 있어 보이지는 않습니다. 어찌되었건, 모델이 간단한 질문에 제대로 예측하지 못하는 것으로 보입니다. 성능을 좀 더 잘 이해하기 위해서, validation set으로 테스트를 해 봅시다.

```shell
$ ./fasttext test model_cooking.bin cooking.valid
N	3000
P@1	0.144
R@1	0.0623
Number of examples: 3000
```

위의 결과는 precision at one(p@1)과 recall at one(r@1) 결과 입니다. 마찬가지로 precision at 5 의 결과와 recall at 5의 결과는 다음과 같이 구할 수 있습니다.

```shell
$ ./fasttext test model_cooking.bin cooking.valid 5
N	3000
P@5	0.0677
R@5	0.146
Number of examples: 3000
```

### Precision and recall
***
precision은 fastText에 의해 예측된 label 중 맞는 label의 비율을 의미합니다. recall은 실제 해당 label들 중에 모델이 맞게 예측한 label의 비율을 의미합니다. 예시를 통해 살펴봅시다.
> Why not put kives in the dishwasher?


Stack Exchange 에서, 이와 같은 문장은 "***equipment***", "***cleaning***", "***knives***" 세개의 tag를 가집니다. 모델에 의해 예측된 5개의 label은 다음과 같습니다.

```shell
$ ./fasttext predict model_cooking.bin - 5
Why not put knives in the dishwasher?
__label__food-safety __label__baking __label__equipment __label__bread __label__substitutions
```

위와 같이 5개의 label ("*food-safety*", "*baking*", "*equipment*", "*substitutions*", "*bread*")을 예측했습니다.<br>
따라서, 모델에 의해 예측된 5개의 label 중 하나 ("***equipment***")가 정확히 예측 되었으며, 이 경우 precision@5은 0.20 입니다. 반면, 3개의 실제 label 중 모델이 정확히 예측한 것은 하나 뿐이며, 따라서 recall@5는 0.333 입니다.

보다 자세한 내용은 다음의 [위키피디아](https://en.wikipedia.org/wiki/Precision_and_recall)를 참고하실 수 있습니다.

## Making the model better
***
기본 arguments로 수행된 fastText 모델은 새로운 질문을 분류하는 데에 좋은 성능을 보이지 못했습니다. paramenters를 수정해가며 모델의 성능을 향상시켜 봅시다.

### Preprocessing the data

데이터를 살펴보면, 일부 단어들은 대문자를 포함하고 있거나 문장부호를 포함하고 있습니다. 모델 성능을 향상시키기 위한 첫 번째 단계는 간단한 pre-processing을 수행하는 것입니다.<br>
간단한 정규화(crude normalization)은 ***sed*** 와 ***tr*** 명령어로 수행할 수 있습니다.

```shell
$ cat cooking.stackexchange.txt | sed -e "s/([.!?,'/()])/ 1 /g" | tr "[:upper:]" "[:lower:]" > cooking.preprocessed.txt
$ head -n 12404 cooking.preprocessed.txt > cooking.train
$ tail -n 3000 cooking.preprocessed.txt > cooking.valid
```

pre-processing을 거친 새로운 데이터로 모델을 학습시켜 봅시다.

```shell
$ ./fasttext supervised -input cooking.train -output model_cooking
Read 0M words
Number of words:  12404
Number of labels: 735
Progress: 100.0%  words/sec/thread: 51206  lr: 0.000000  loss: 9.910273  eta: 0h0m
```

```shell
$ ./fasttext test model_cooking.bin cooking.valid
N	3000
P@1	0.165
R@1	0.0715
Number of examples: 3000
```

pre-processing 단계 덕분에, 같은 데이터로 학습된 단어의 수가 줄어들었고(14K to 12K), precision역시 약 2%가량 증가한 것을 확인할 수 있다.

### More epochs and larger learning rate
기본적으로, fastText는 training examples를 체크하는 epoch가 5번으로 설정되어있습니다. 튜토리얼의 training set에는 12K개의 example이 있기 때문에 매우 작은 숫자 압니다. 전체 데이터를 체크하는 횟수(epoch라고 함)를 늘리기 위해서 "***-epoch***" 옵션을 사용할 수 있습니다.

```shell
$ ./fasttext supervised -input cooking.train -output model_cooking -epoch 25
Read 0M words
Number of words:  12404
Number of labels: 735
Progress: 100.0%  words/sec/thread: 50291  lr: 0.000000  loss: 7.439700  eta: 0h0m h-14m
```

모델을 테스트 해 봅시다.

```shell
$ ./fasttext test model_cooking.bin cooking.valid                                        
N  3000
P@1  0.501
R@1  0.218
Number of examples: 3000
```

훨씬 더 좋아졌습니다! 우리 모델의 학습 속도를 변경하는 방법 중 하나는 learning rate를 올리거나 낮추는 것입니다. 이것은 각 example을 수행한 수 얼마나 많이 변경될 것인지를 결정합니다. learning rate가 0인 모델은 모델이 변하지 않는다는 것을 의미하고, 아무것도 학습하지 못합니다. learning rate의 추천하는 범위는 0.1~1.0 사이 입니다.

```shell
$ ./fasttext supervised -input cooking.train -output model_cooking -lr 1.0
Read 0M words
Number of words:  12404
Number of labels: 735
Progress: 100.0%  words/sec/thread: 49606  lr: 0.000000  loss: 6.757581  eta: 0h0m
```

```shell
$ ./fasttext test model_cooking.bin cooking.valid                         
N  3000
P@1  0.563
R@1  0.245
Number of examples: 3000
```

더 나아졌습니다 ! 이제 둘 다 사용해 보겠습니다.

```shell
$ ./fasttext supervised -input cooking.train -output model_cooking -lr 1.0 -epoch 25
Read 0M words
Number of words:  12404
Number of labels: 735
Progress: 100.0%  words/sec/thread: 49754  lr: 0.000000  loss: 4.482828  eta: 0h0m
```

```shell
$ ./fasttext test model_cooking.bin cooking.valid                                   
N  3000
P@1  0.585
R@1  0.255
Number of examples: 3000
```

이제 성능을 더욱 향상시키기 위해 새로운 feature들을 추가해 보겠습니다.

### word n-grams
마지막으로, unigram 대신 bigram이라는 단어를 사용하여 모델의 성능을 향상시킬 수 있습니다. 이는 감성분석과 같이 단어의 순서가 분류하는데 있어 중요한 역할을 할때 특히 더 영향을 끼칩니다.

```shell
$ ./fasttext supervised -input cooking.train -output model_cooking -lr 1.0 -epoch 25 -wordNgrams 2
Read 0M words
Number of words:  12404
Number of labels: 735
Progress: 100.0%  words/sec/thread: 48367  lr: 0.000000  loss: 3.210917  eta: 0h0m
```

```shell
$ ./fasttext test model_cooking.bin cooking.valid
N	3000
P@1	0.588
R@1	0.254
Number of examples: 3000
```

몇가지 단계를 거쳐서 precision@1을 14.4% 에서 58.8%까지 끌어올렸습니다. 주요 단계는 다음과 같았습니다.
- 데이터 pre-procession
- epoch 수 변경("***-epoch***" 옵션 활용, 표준 범위 [5 - 50])
- learning rate 변경("***-lr***" 옵션 활용, 표준 범위 [0.1 - 1.0])
- word n-gram 활용("***-wordNgrams***" 옵션 활용, 표준 범위 [1-5])

### What is a Bigram?
"***unigram***"은 일반적으로 모델의 input으로 활용되는 분할되지 않은 하나의 unit 혹은 token을 나타냅니다. 예를 들어, unigram은 모델에 따라 하나의 단어가 될 수도 있고 하나의 문자가 될 수도 있습니다. fastText 에서는, 단어 수준에서 동작하므로 여기서의 unigrams는 단어 입니다.

이와 비슷하게, "***bigram***"은 연속적인 2개의 token 또는 단어의 연결을 의미합니다. 마찬가지로, n개의 연속적인 token 또는 단어의 연결을 "***n-gram***"이라고 표현합니다.

예를 들어, 다음의 문장에서,
> "*Last donut of the night*"

***unigram*** 은 'Last', 'donut', 'of', 'the', 'night'입니다.<br>
반면, ***bigram*** 은 'Last dount', 'donut of', 'of the', 'the night'가 됩니다.

***bigram*** 은 흥미로운 점이 있습니다. 대부분의 문장에서 bag of n-gram을 보고 단어를 재구성할 수 있기 때문입니다.

다음과 같이 주어진 bag of bigram을 통해 문장을 재구성해 보면 재미있을 것 같습니다.
> "all out", "I am", "of bubblegum", "out of", "am all"

여기서 마찬가지로, 한 단어를 unigram이라 부르는 것이 일반적입니다.

### Scaling things up
***
우리가 수천개의 example들로 모델을 학습하고 있기 때문에, 훈련이 그렇게 오래 걸리지는 않습니다. 그러나 모델을 학습시키는데 사용하는 데이터셋이 더 커지고 label이 더 많아지면 매우 느려질 수 있습니다. 훈련을 보다 빨리 수행할 수 있는 방법은 일반적인 soft max대신 hierachical softmax를 사용하는 것입니다. 이는 "***-loss hs***" 옵션을 통해 수행될 수 있습니다.

```shell
$ ./fasttext supervised -input cooking.train -output model_cooking -lr 1.0 -epoch 25 -wordNgrams 2 -bucket 200000 -dim 50 -loss hs
Read 0M words
Number of words:  12404
Number of labels: 735
Progress: 100.0%  words/sec/thread: 1624606  lr: 0.000000  loss: 2.344548  eta: 0h0m
```

이 예제에서 위와 같이 수행하면, 모델 학습이 수 초 이내에 끝날 정도로 빨라지는 것을 확인할 수 있습니다.

## Conclusion
***
이 튜토리얼에서는 강력한 text classifier를 학습시키기 위해 fastText를 활용하는 방법에 대하여 알아보았습니다. 또한 model tunning시 활용할 수 있는 중요한 옵션들에 대하여서도 간략히 살펴보았습니다.<br>
다음 예제에서는 Word vector를 생성하는 방법에 대하여 알아보겠습니다.

***
번역 및 수정 : Hyunlim YANG (lims1@dgist.ac.kr), InfoLab., DGIST / 2017.06.20
***
