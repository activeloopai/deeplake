<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/activeloopai/Hub/master/docs/logo/logo-explainer-bg.png" width="50%"/>
    </br>
</p>
<p align="center">
    <a href="http://docs.activeloop.ai/">
        <img alt="Docs" src="https://readthedocs.org/projects/hubdb/badge/?version=latest">
    </a>
    <a href="https://pypi.org/project/hub/"><img src="https://badge.fury.io/py/hub.svg" alt="PyPI version" height="18"></a>
    <a href="https://pepy.tech/project/hub"><img src="https://static.pepy.tech/personalized-badge/hub?period=month&units=international_system&left_color=grey&right_color=orange&left_text=Downloads" alt="PyPI version" height="18"></a>
    <a href="https://app.circleci.com/pipelines/github/activeloopai/Hub">
    <img alt="CircleCI" src="https://img.shields.io/circleci/build/github/activeloopai/Hub?logo=circleci"> </a>
     <a href="https://github.com/activeloopai/Hub/issues">
    <img alt="GitHub issues" src="https://img.shields.io/github/issues/activeloopai/Hub"> </a>
    <a href="https://codecov.io/gh/activeloopai/Hub/branch/master"><img src="https://codecov.io/gh/activeloopai/Hub/branch/master/graph/badge.svg" alt="codecov" height="18"></a>
    <a href="https://twitter.com/intent/tweet?text=The%20fastest%20way%20to%20access%20and%20manage%20PyTorch%20and%20Tensorflow%20datasets%20is%20open-source&url=https://activeloop.ai/&via=activeloopai&hashtags=opensource,pytorch,tensorflow,data,datascience,datapipelines,activeloop,dockerhubfordatasets"> 
        <img alt="tweet" src="https://img.shields.io/twitter/url/http/shields.io.svg?style=social"> </a>  
   </br> 
    <a href="https://join.slack.com/t/hubdb/shared_invite/zt-ivhsj8sz-GWv9c5FLBDVw8vn~sxRKqQ">
  <img src="https://user-images.githubusercontent.com/13848158/97266254-9532b000-1841-11eb-8b06-ed73e99c2e5f.png" height="35" /> </a>
    <a href="https://docs.activeloop.ai/en/latest/?utm_source=github&utm_medium=readme&utm_campaign=button">
  <img src="https://i.ibb.co/YBTCcJc/output-onlinepngtools.png" height="35" /></a>

---

</a>
</p>

<h3 align="center"> Представляем Data 2.0, при поддержке Hub. </br>Самый быстрый способ организации хранения, доступа и работы с датасетами с помощью контроля версий для работы с PyTorch/TensorFlow. Работает как локально, так и в облаке. Гибкие дата-пайплайны.</h3>

---

[ English | [Français](./README_translations/README_FR.md) | [简体中文](./README_translations/README_CN.md) | [Türkçe](./README_translations/README_TR.md) | [한글](./README_translations/README_KR.md) | [Bahasa Indonesia](./README_translations/README_ID.md)] | [Русский](./README_translations/README_RU.md)]

Примечание: данный перевод может обновляться не одновременно с основным. Для прочтения актуальной версии обращайтесь, пожалуйста, к README на английском языке. </i>

### Для чего нужен Hub?

Software 2.0 нуждается в Data 2.0, это обеспечивает Hub. Основную часть времени разработчики в области Data Science и ML занимаются обработкой и предподготовкой данных вместо тренировки моделей. С помощью Hub мы решаем эту проблему. Мы храним ваши датасеты (даже размером в петабайты данных) в виде отдельного numpy-подобного массива в облаке, поэтому вы можете легко получить их и работать на любом устройстве. Hub делает любой тип данных (графика, некстовые файлы, аудио или видео), хранимых в облаке, доступным, словно он хранится локально. С одинаковой версией датасета, работа вашей команды всегда будет синхронизирована. 

Hub используется такими организациями как Waymo, Red Cross, World Resources Institute, Omdena, а также многих других.

### Основные характеристики 

* Сохраняйте и скачивайте огромные датасеты, используя контроль версий
* Работайте совместно как в Google Docs: Множество разработчиков работают с одноми и теми же синхронизированными данными, не мешая друг другу
* Одновременный доступ с множества различных устройств
* Выполнение деплоя кудв угодно - локально, в Google Cloud, S3, Azure, а также в Activeloop (по умолчанию и совершенно бесплатно!) 
* Интегрируйте с такими вашими ML-инструментами как Numpy, Dask, Ray, [PyTorch](https://docs.activeloop.ai/en/latest/integrations/pytorch.html), or [TensorFlow](https://docs.activeloop.ai/en/latest/integrations/tensorflow.html)
* Создавайте настолько большие массивы, насколько пожелаете. Вы можете хранить графические изображения размером до 100k на 100k!
* Храните размер каждого образца динамически. Таким образом вы можете хранить большие и маленькие массивы в виде одного массива. 
* [Визуализация](http://app.activeloop.ai/?utm_source=github&utm_medium=repo&utm_campaign=readme) любая часть данных за секунды без лишних манипуляций

 <p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/activeloopai/Hub/master/docs/visualizer%20gif.gif" width="75%"/>
    </br>
Визуализация загрузки датасета в Hub с помощью app.activeloop.ai (бесплатный инстрмент).

</p>


## Начало работы
Работайте с доступными датасетами или со своим собственным локально или в любом облачном хранилище.

### Получение открытых данных. Быстро.

Чтобы плучить открытй датасет, необходимо писать десятки строк кода и тратить часы на получение доступа к API и его изучению, а также на загрузку данных. С помощью Hub потребуется всего 2 строчки кода и вы **можете приступать к работе с датасетом втечение 3 минут**.

```sh
pip3 install hub
```

Получайте доступ к публичным датасетам в Hub, следуя простым инструкциям и используя несколько строк кода. Запустите данный кусок кода чтобы получить первую тысячу изображений из [датасета MNIST](https://app.activeloop.ai/dataset/activeloop/mnist/?utm_source=github&utm_medium=repo&utm_campaign=readme) в виде numpy-массива:
```python
from hub import Dataset

mnist = Dataset("activeloop/mnist")  # ленивая загрузка данных MNIST
# экономьте время с помощью *compute* чтобы получить только нужные данные
mnist["image"][0:1000].compute()
```
Вы можете найти остальные популярные датасеты на [app.activeloop.ai](https://app.activeloop.ai/datasets/popular/?utm_source=github&utm_medium=repo&utm_campaign=readme).

### Обучение модели

Загрузиите данные и обучите модель **напрямую**. Hub имеет интеграцию с PyTorch и TensorFlow, а также выполняет конвертацию между форматами в понятном виде. Посмотрите на пример с PyTorch ниже:

```python
from hub import Dataset
import torch

mnist = Dataset("activeloop/mnist")
# конвертация MNIST в формат PyTorch
mnist = mnist.to_pytorch(lambda x: (x["image"], x["label"]))

train_loader = torch.utils.data.DataLoader(mnist, batch_size=1, num_workers=0)

for image, label in train_loader:
    # Далее цикл обучения
```

### Создание датасета локально 
Если вы хотите работать со своими данными локально, вы можете начать с создания датасета:
```python
from hub import Dataset, schema
import numpy as np

ds = Dataset(
    "./data/dataset_name",  # путь к датасету
    shape = (4,),  # следует конвенции numpy shape
    mode = "w+",  # режим чтения и записи
    schema = {  # именованные части данных, которые могут определять формат
    # Тензор является сгенерированной структурой, которая может хранить любой тип данных
        "image": schema.Tensor((512, 512), dtype="float"),
        "label": schema.Tensor((512, 512), dtype="float"),
    }
)

# заполнение данных (в данном случае - нулями для инициализации)
ds["image"][:] = np.zeros((4, 512, 512))
ds["label"][:] = np.zeros((4, 512, 512))
ds.flush()  # вызов создания датасета
```

Вы также можете указать `s3://bucket/path`, `gcs://bucket/path` или путь на azure. [Здесь](https://docs.activeloop.ai/en/latest/simple.html#data-storage) вы можете найти больше информации по хранению в облаке.
Также если вам нужен публичный датасет и вы не можете найти его в Hub, вы можете оставить [заявку](https://github.com/activeloopai/Hub/issues/new?assignees=&labels=i%3A+enhancement%2C+i%3A+needs+triage&template=feature_request.md&title=[FEATURE]+New+Dataset+Required%3A+%2Adataset_name%2A). Мы сделаем его доступным для всех как только сможем!

### Загружайте свой датасет и получайте доступ к нему <ins>из любого места</ins> в 3 простых шага

1. Пройдите бесплатную регистрацию в [Activeloop](https://app.activeloop.ai/register/?utm_source=github&utm_medium=repo&utm_campaign=readme) и авторизируйтесь локально:
    ```sh
    hub register
    hub login

    # При необходимости, укажите username и password в качестве аргументов (используется на таких платформах как Kaggle)
    hub login -u username -p password
    ```
    В будущих релизах появится команда `activeloop`. Ниже представлен её синтаксис для использования:
    ```sh
    activeloop register
    activeloop login

    # При необходимости, укажите username и password в качестве аргументов (используется на таких платформах как Kaggle)
    activeloop login -u username -p password
    ```

2. Затем создайте датасет, укажите его название и загрузите в ваш аккаунт. Например:
    ```python
    from hub import Dataset, schema
    import numpy as np

    ds = Dataset(
        "username/dataset_name",
        shape = (4,),
        mode = "w+",
        schema = {
            "image": schema.Tensor((512, 512), dtype="float"),
            "label": schema.Tensor((512, 512), dtype="float"),
        }
    )

    ds["image"][:] = np.zeros((4, 512, 512))
    ds["label"][:] = np.zeros((4, 512, 512))
    ds.flush()
    ```

3. Получайте доступ к нему где угодно, на любом устройстве, где есть командная строка:
    ```python
    from hub import Dataset

    ds = Dataset("username/dataset_name")
    ```


## Документация

Для таких более продвинутых работ с данными как загрузка больших датасетов или выполнения различных преобразований, пожалйста обращайтесь к [документации](http://docs.activeloop.ai/?utm_source=github&utm_medium=repo&utm_campaign=readme).

## Примеры в Notebooks
[Данный каталог](https://github.com/activeloopai/Hub/tree/master/examples) хранит ряд примеров и [этот каталог](https://github.com/activeloopai/Hub/tree/master/examples/notebooks) примеры в ноутбуках. Некоторые ноутбуки представлены ниже.

| Notebook  	|   Описание	|   	|
|:---	|:---	|---:	|
| [Загрузка изображений](https://github.com/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%201a%20-%20Uploading%20Images.ipynb) | Показывает как загружать и хранить изображения в Hub |  [![Открыть в Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%201a%20-%20Uploading%20Images.ipynb) |
| [Загрузка датафреймов](https://github.com/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%201b%20-%20Uploading%20Dataframes.ipynb)  	| Показывает как загружать датафреймы в Hub  	| [![Открыть в Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%201b%20-%20Uploading%20Dataframes.ipynb)  	|
| [Загрузка Audio](https://github.com/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%201c%20-%20Uploading%20Audio.ipynb) | Объясняет как работать с аудио в Hub|[![Открыть в Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%201c%20-%20Uploading%20Audio.ipynb) |
| [Скачивание данных](https://github.com/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%202%20-%20Retrieving%20Remote%20Data.ipynb) | Объясняет как скачивать данные в Hub| [![Открыть в Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/activeloopai/Hub/blob/master/examples/tutorial/tutorial/Tutorial%202%20-%20Retrieving%20Remote%20Data.ipynb) |
| [Преобразование данных](https://github.com/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%203%20-%20Transforming%20Data.ipynb) | Краткое описание преобразований данных в Hub|[![Открыть в Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%203%20-%20Transforming%20Data.ipynb) |
| [Динамические тензоры](https://github.com/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%204%20-%20What%20are%20Dynamic%20Tensors.ipynb) | Работы с данными различной размерности в Hub|[![Открыть в Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%204%20-%20What%20are%20Dynamic%20Tensors.ipynb) |
| [NLP с помощью Hub](https://github.com/activeloopai/Hub/blob/master/examples/notebooks/nlp_using_hub.ipynb) | Настройка Bert для CoLA|[![Открыть в Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/activeloopai/Hub/blob/master/examples/notebooks/nlp_using_hub.ipynb) |


## Варианты использования
* **Снимки со спитников и дронов**: [Интеллектуальное сельское хозяйство с масштабируемыми воздушными трубопроводами](https://activeloop.ai/usecase/intelinair?utm_source=github&utm_medium=repo&utm_campaign=readme), [Составление карты экономического благополучия в Индии](https://towardsdatascience.com/faster-machine-learning-using-hub-by-activeloop-4ffb3420c005), [Борьба с пустынной саранчой в Кении с помощью Красного Креста](https://omdena.com/projects/ai-desert-locust/)
* **Медицинские снимки**: Объемные изображения, такие как МРТ или рентгеновский снимок
* **Самоходные автомобили**: [Radar, 3D LIDAR, Point Cloud, Semantic Segmentation, Video Objects](https://medium.com/snarkhub/extending-snark-hub-capabilities-to-handle-waymo-open-dataset-4dc7b7d8ab35)
* **Розничная торговля**: Наборы данных для самообслуживания
* **Медиа**: Хранение изображений, видео, аудио

## Что особенного в Hub?

Существует довольно много библиотек управления наборами данных, которые предлагают функции, которые могут показаться похожими на Hub. Фактически, довольно много пользователей переносят данные из наборов данных PyTorch или Tensorflow в Hub. Вот несколько поразительных отличий, с которыми вы столкнетесь после перехода на Hub:
* данные предоставляются фрагментами, которые вы можете передавать из удаленного места, вместо того, чтобы загружать все сразу
* поскольку оценивается только необходимая часть набора данных, вы можете сразу же работать с данными
* вы можете хранить данные, которые не поместятся в вашей памяти целиком
* вы можете управлять версиями и сотрудничать с несколькими пользователями над вашими наборами данных на разных машинах
* у вас есть инструменты, которые за секунды улучшают ваше понимание данных, например, наш инструмент визуализации
* вы можете легко подготовить свои данные для нескольких обучающих библиотек одновременно (например, вы можете использовать один и тот же набор данных для обучения с PyTorch и Tensorflow)


## Сообщество

Присоединяйтесь к нашему [**Slack сообществу**](https://join.slack.com/t/hubdb/shared_invite/zt-ivhsj8sz-GWv9c5FLBDVw8vn~sxRKqQ) чтобы получить помощь от команды Activeloop и других пользователей, а также быть в курсе лучших практик управления наборами данных и предварительной обработки.

<img alt="tweet" src="https://img.shields.io/twitter/follow/activeloopai?label=stay%20in%20the%20Loop&style=social"> в Twitter.

Как всегда, спасибо нашим замечательным участникам!    

<a href="https://github.com/activeloopai/hub/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=activeloopai/hub" />
</a>

Сделано с использованием [contributors-img](https://contrib.rocks).

Пожалуйста прочтайте [CONTRIBUTING.md](CONTRIBUTING.md) чтобы узнать, как принимать участие в развитии Hub.

## Примеры
Формат Activeloop Hub позволяет добиться более быстрого вывода с меньшими затратами. У нас уже есть 30+ популярных наборов данных на нашей платформе. К ним относятся:
- COCO
- CIFAR-10
- PASCAL VOC
- Cars196
- KITTI
- EuroSAT 
- Caltech-UCSD Birds 200
- Food101

Проверьте эти и многие другие популярные наборы данных в нашем [веб-приложении визуализаторе](https://app.activeloop.ai/datasets/popular/?utm_source=github&utm_medium=repo&utm_campaign=readme) и загрузите их напрямую для обучения модели!

## README значок

Используете Hub? Добавьте README значок, чтобы все могли об этом узнать: 


[![hub](https://img.shields.io/badge/powered%20by-hub%20-ff5a1f.svg)](https://github.com/activeloopai/Hub)

```
[![hub](https://img.shields.io/badge/powered%20by-hub%20-ff5a1f.svg)](https://github.com/activeloopai/Hub)
```

## Отслеживание использования
По умолчанию мы собираем анонимные данные по использованию нашего продукта с помощью Bugout (вот [код](https://github.com/activeloopai/Hub/blob/853456a314b4fb5623c936c825601097b0685119/hub/__init__.py#L24), который это выполняет). Он регистрирует только собственные действия и параметры библиотеки Hub. Данные о пользователях и моделях не собираются.

Это помогает команде Activeloop понять, как используется инструмент и как принести максимальную пользу сообществу, создавая действительно полезный функцонал. Вы легко можете отказать от отслеживания использования во время своей авторизации.

## Дисклеймер

По аналогии с другими системами управления датасетами, `Hub` это служебная библиотека, которая загружает и подготавливает общедоступные наборы данных. Мы не размещаем и не распространяем эти наборы данных, не ручаемся за их качество или точность и не заявляем, что у вас есть лицензия на использование набора данных. Вы обязаны определить, есть ли у вас разрешение на использование датасета в соответствии с лицензией на датасет.


Если вы являетесь владельцем датасета и хотите обновить какую-либо его часть (описание, цитату и т.д.) или не хотите, чтобы ваш набор данных был включен в эту библиотеку, свяжитесь с нами через [issue на GitHub](https://github.com/activeloopai/Hub/issues/new). Спасибо за ваш вклад в сообщество машинного обучения!


## Благодарности
 Эта технология была вдохновлена нашим опытом в Принстонском университете, и я хотел бы поблагодарить Уильяма Сильверсмита @SeungLab за его потрясающий инструмент [cloud-volume](https://github.com/seung-lab/cloud-volume). Мы активно используем [Zarr](https://zarr.readthedocs.io/en/stable/) и хотели бы поблагодарить их сообщество за создание такого замечательного фундаментального блока.
