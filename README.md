# Report

## 0. Дизайн эксперимента.

[`src/create_annotation.py`](src/create_annotation.py)

[`src/create_design.py`](src/create_design.py)

У меня есть ~1400 картин в 8 стилях. В названиях файлов некоторых стилей сохранились имена авторов (авторов нет у фотографий и мультиков). Я решил разбить датасет на части таким образом:
- 5% отложенный валидационный датасет
- оставшиеся 95% = 4 фолда для CV

Я сделал так, что все (4 + 1) = 5 фолдов не имеют общих авторов. Это для того, чтобы снизить переобучение / data leak. Стиль одного автора может быть запоминающимся, а поэтому классификатор, тренирующийся на картинах этого автора, вероятно, сможет отгадать другие его картины в тестовой выборке. Разбивая по авторам, я старался делать так, чтобы во все фолды попало примерно одинаковое число авторов и одновременно с этим примерно одинаковое число картин.

В отличие от 4 CV фолдов, валидационный датасет дополнительно усложнён -- я сделад в нём как можно больше авторов. Так, в него попало много авторов с всего одной картиной. Таблица с числом авторов и картин в каждом фолде:

```
              author                 filename                
fold_author        0   1   2   3 val        0   1   2   3 val
label                                                        
artdeco            2   2   2   2   3       10  31  10  11   3
cartoon            1   1   1   1   1       18  18  18  17   4
cubism            17  17  17  16  18       81  89  85  92  18
impressionism     23  23  23  22  11       58  50  61  60  11
japonism           6   6   5   5   7       55  42  57  40   9
naturalism         2   2   2   2   3       45  83  61  20   9
photo              1   1   1   1   1       33  33  33  33   7
rococo             5   5   4   4   5       20  21  47  24   5
```

Метрикой качества для всех задач классификации я избрал top 2 accuracy, которую взвешивал по классам. (Ещё много метрик я просто логгировал, но не опирался на них при выборе модели, см. [`src/scorers/ScorerCombo.py`](src/scorers/ScorerCombo.py)). Top 2 accuracy мне кажется более подходящей, чем обычная accuracy (top 1) или f1, потому что классы, на самом деле, не взаимоисключающие и отбрасывать модель за то, что она нашла пару правдаподобных классов вместо одного, я не хочу. Например, можно найти картину, на которой нарисованы животные яркими красками, широкими мазками. Это, наверное, частично и натурализм, и импрессионизм. Более того, сами стили в истории не возникали внезапно. У каждого художника есть что-то от "соседей". Впрочем, какие-то классы действительно сильно отличаются от всех остальных -- фото, например. Top 3 accuracy я не беру, потому что 3 это уже почти половина моих 8 классов.

## 1. Классификация картин по стилям с помощью нейросети.

### Архитектура

![model-design](images/model-design.svg)

[`src/models/models.py`](src/models/models.py)

Я собрал нейросеть, как на картинке выше.
1. Resnet генерирует признаки и отдаёт 1000-мерный вектор через ReLU в полносвязный слой. То, что вышло из этого слоя, я называю embedding, а всю конструцию, которая его создаёт -- Embedder.
2. Embedding отправляется ещё через одну ReLU в последний полносвязный слой, который на выходе даёт logits для 8 классов.

Я взял CrossEntropyLoss, как функцию ошибок для своей задачи.

Картинки в трейне подвергались таким трансформациям:
```python
transform_resnet = resnet_weights.DEFAULT.transforms()
transform_train = transforms.Compose(
    [
        transforms.RandomRotation(45),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.25, contrast=0.25, saturation=0.25, hue=0.0
        ),
        transforms.RandomErasing(p=0.5),
        transform_resnet,
    ]
)
```

В тесте использовался только `transform_resnet`.

### Оптимизация гимерпараметров
[`src/optunize_nn_classifier.py`](src/optunize_nn_classifier.py)

Гиперпараметры, которые я перебирал при обучении зафиксированы в [params.yaml](params.yaml), раздел `optunize_nn_classifier`. Главные из них:
- Глубина ResNet. В моих экспериментах CV, качество было тем выше, чем глубже ResNet. Но выйгрыш был незначительным, например, median top 2 accuracy по 4 фолдам: `0.76` и `0.81` для ResNet18 и ResNet50, соответственно. Расчеты с очень глубокими моделями сильно растягиваются по времени (я считал всё на apple m1 pro), поэтому для всех дальнейших экспериментов я пользовался ResNet18.
- Так же из CV я обнаружил, что стоит разморозить весь ResNet и обучать его вместе новыми с добавленными слоями.
- Learning rate. Так как весь ResNet уже сидит в хорошем минимуме лосса, то темп обучения я делал изначально маленький плюс экспоненциально уменьшал его к концу расчета до `1e-5`.
- Embedding size. Тут проклятие размерности нам на руку, потому что разнести классы в многомерном пространстве значительно легче, чем в маломерном.

Ещё я подготовил для кручения label_smoothing и weight_decay, но так и не трогал их, хотя они могли бы улучшить генерализацию модели. Так же, стоило сделать два оптимизатора, которые работают одновременно. Один оптимизирует CNN (feature generator), другой -- линейные слои. Первый должен мелко шагать, а второй крупно.

По итогам кроссвалидации я выбрал такие параметры:

```yaml
mlflow_run_id: ca8ff65afb5744a2bde060f116d70a22
resnet_name: resnet18
embedding_size: 16
lr_start: 0.0001
```

На этих параметрах я обучил финальную модель, выбрав такое разбиение на train и test (из 4 уже приготовленных фолдов), которое давало лучшее качество.

### Результаты обучения лучшей модели

![learning-curves](images/learning-curves-ca8ff65afb5744a2bde060f116d70a22.png)

Confusion matrix. Нормализована по строкам (на главной диагонали получается recall).
![confmat](images/confmat-ca8ff65afb5744a2bde060f116d70a22.png)

Classification report.
```
               precision    recall  f1-score   support

      artdeco       0.30      0.70      0.42        10
      cartoon       0.92      0.67      0.77        18
       cubism       0.91      0.80      0.85        85
impressionism       0.83      0.70      0.76        61
     japonism       0.65      0.81      0.72        57
   naturalism       0.71      0.64      0.67        61
        photo       0.88      0.91      0.90        33
       rococo       0.83      0.87      0.85        46

     accuracy                           0.77       371
    macro avg       0.75      0.76      0.74       371
 weighted avg       0.79      0.77      0.78       371
```

#### artdeco
У artdeco проблемы с precision. Если для "продуктовой" модели это будет критично, то нужно обучить отдельную модель, бинарный классификатор artdeco vs all. Для этой задачи стоит пересобрать выборку (проредить остальные классы), так как artdeco на порядок менее представлен. Можно выбрать более подходящую функцию ошибок, например, FocalLoss.

#### naturalism
Лучше artdeco, но всё равно плохо. Naturalism достаточно представлен в датасете. Проблема с ним в другом. Этот класс скорее не класс, а "атрибут" или "label" (multi-label task).

#### photo
Это чемпион: f1_score = 0.9. Здесь всё понятно, класс photo не является живописным.

### Валидация
В самом конце, я достаю val датасет и проверяю финальное качество -- top 2 accuracy = 0.8.
За бейзлайн можно взять случайное угадывание -- оно даёт 0.25.
```
       precision  recall  fscore  top_1_accuracy  top_2_accuracy
name                                                            
train       0.96    0.96    0.96            0.97            0.99
test        0.79    0.77    0.77            0.76            0.91
val         0.63    0.62    0.61            0.63         -> 0.80
```

### Примеры

Top-1
![examples-top-1](images/prediction-examples-top-1-ca8ff65afb5744a2bde060f116d70a22.png)

Top-2
![examples-top-2](images/prediction-examples-top-2-ca8ff65afb5744a2bde060f116d70a22.png)

Wrong
![examples-wrong](images/prediction-examples-wrong-ca8ff65afb5744a2bde060f116d70a22.png)

### XAI?

Интересно сделать occlusion для определения важных участков картин.

## 2. Кластеризация эмбеддингов (выходы предпоследнего слоя).

На картинке показана "эволюция" эмбеддингов. На 9 эпохе модель достигла лучшего качества.
![embedding-evolution](images/embedding-evolution-ca8ff65afb5744a2bde060f116d70a22.png)

Кластеризация с помощью и kNN, и AgglomerativeClustering дала максимальный silhouette score при числе кластеров равном числу классов (8). Следующая картинка получена на данных с лучшей эпохи:

![embedding-best-epoch](images/embedding-clustering-ca8ff65afb5744a2bde060f116d70a22.png)

Japonism, cartoon и photo сидят в своих отдалённых кластерах (artdeco едва соединяется с cubism).
С cartoon и photo понятно -- это не картины.
Japonism отделился, вероятно, потому что не является европейским живописным стилем.
Остальные классы расположены интереснее.
Я вижу путь от rococo через naturalism, impressionism и cubism к artdeco.
Про naturalism я не уверен (он, кстати, на impressionism наполз), но остальные классы располагаются в хронологическом порядке (если сверяться с Википедией).

*Заметка. Так же, в ходе экспериментов я заметил, что при увеличении learning rate на порядок такой красивой кластеризации не получается, хотя задача классификации решается примерно так же хорошо.*

## 3. Metric learning

[`src/train_embedder.py`](src/train_embedder.py)

Metric learning хорошо подходит для open-set задач.
В моём случае, финальная цель -- классифицировать уже сформированные, зафиксированные классы (closed-set).
Впрочем, можно попробовать построить эмбединги с помощью metric learning на всех, кроме, скажем, пары классов.
После обучения заэмбеддить отложенные классы и оценить результат.
Отложил я классы photo и naturalism.

Мой опыт подсказывает, что функции ошибки, работающие с углами через CosineSimilarity, проявляют себя лучше всего при построении эмбедингов.
Я взял ArcFaceLoss, но даже на нём задача решалась плохо.
В качестве метрики я использовал (как в предыдущем пункте про кластеризацию) silhouette score.
Так я сделал, потому что это просто и быстро реализовать.

Так как ArcFaceLoss работает с эмбеддингами на гиперсфере, я решил сделать эмбеддер в 3D, чтобы рисовать занимательные картинки. В многомерном пространстве я тоже строил эмбеддинги, но качество от этого не менялось.

Нативное пространство: [click](images/arccos-native-60721c74bc834d52a53f486ff813d08c.html).

Нормализованные векторы эмбеддингов (лежат на единичной сфере): [click](images/arccos-sphere-60721c74bc834d52a53f486ff813d08c.html).

Отложенные классы размазались по всей поверхности сферы, хотя если их убрать, то видно, что модель хорошо научилась разносить оставшиеся классы из обучающей выборки.

*Так же я сделал следующий эксперимент. Для первой задачи классификации я применил два лосса одновременно: TripletMarginLoss для эмбеддингов и CrossEntropy для logits. Это привело у ухудшению качества классификации. Результаты не показываю.*

## 4. Обучение Random Forest на эмбеддингах.

[`src/optunize_random_forest.py`](src/optunize_random_forest.py)

Полученный эмбеддинг я использовал, как признаки для обучения Random Forest (его я выбрал для простоты). Учил на такой же разбивке трейн/тест, как в пункте 1. Привлёк optuna для оптимизации гиперпараметров. Качество получается хуже, чем у нейросети.

```
        top_2_accuracy

        NN      random_forest
                                                         
train   0.99    0.96
test    0.91    0.86
val     0.80    0.77
```

Это разумно, несмотря на то, что в нейросети остаётся только один линейный слой, который делает из эмбеддинга финальные logits. Разумно потому, что весь хвост из предыдущих слоёв учился сначала на ImageNet, потом на моём датасете, чтобы последний слой мог успешно вычислить logits. 

Random Forest (и подобные ему) алгоритмы будут выигрывать в решении этой задачи в другом сценарии. Например, весь ResNet заморожен вместе со своим выходным полносвязным слоем, а учатся только два линейных слоя в конце. В данном случае, Random Forest может дорасти до состояния, в котором он обгонит в качестве нейросеть.

Остаётся вопрос, если у нас получилось 2 алгоритма:
1. Нейросеть = Эмбеддер + Голова
2. Гибрид = Эмбеддер + Random Forest

Какую комбинацию брать?

Ответ зависит от цели (в конце концов, Random Forest может дать качество чуть лучше), но я бы использовал нейросеть.

Во-первых, её предсказания лучше откалиброваны в смысле вероятности, особенно, если лучшая модель выбиралась не по метрике, а по лоссу. Если нас заинтересуют предсказанные вероятности классов, то Random Forest нужно будет дополнительно калибровать.

https://dl.acm.org/doi/abs/10.1145/1102351.1102430

https://arxiv.org/pdf/1706.04599.pdf
 
Во-вторых, одна нейросеть выглядит элегантнее. (Особенно, если к Random Forest придётся добавить калибровщика).


# Resources

## Models

http://math.lakeforest.edu/banerji/research_files/WCVA16.pdf

http://cs231n.stanford.edu/reports/2017/pdfs/410.pdf

http://cs231n.stanford.edu/reports/2017/pdfs/406.pdf

https://www.sciencedirect.com/science/article/abs/pii/S0957417418304421

## Metric learning

https://github.com/KevinMusgrave/pytorch-metric-learning

https://arxiv.org/pdf/2003.11982.pdf

https://discuss.pytorch.org/t/triplet-vs-cross-entropy-loss-for-multi-label-classification/4480

https://arxiv.org/pdf/1902.09229.pdf

## XAI

https://github.com/kazuto1011/grad-cam-pytorch

https://github.com/marcoancona/DeepExplain

https://github.com/albermax/innvestigate

https://christophm.github.io/interpretable-ml-book/pixel-attribution.html#deconvnet

## Misc

https://arxiv.org/pdf/1609.04836.pdf

https://arxiv.org/pdf/1706.04599.pdf
