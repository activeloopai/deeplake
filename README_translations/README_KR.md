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
    <a href="https://pypi.org/project/hub/"><img src="https://img.shields.io/pypi/dm/hub.svg" alt="PyPI version" height="18"></a>
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

<h3 align="center"> Hub에서 제공하는 Data 2.0을 소개합니다. </br>Hub는 PyTorch/TensorFlow 등의 프레임워크에서 데이터에 접근 & 관리하고 버전관리까지 할 수 있는 가장 빠른 방법이며, 로컬이나 클라우드 어디서든 사용 가능합니다. </br>또한 언제든 확장 가능한 데이터 파이프라인입니다.</h3>

---

[ [English](../README.md) | [Français](./README_FR.md) | [简体中文](./README_CN.md) | [Türkçe](./README_TR.md) | 한글 | [Bahasa Indonesia](./README_ID.md)]

### Hub는 무엇인가요?

소프트웨어 2.0 시대에 맞춰, Hub는 데이터 2.0을 제공합니다. 대부분의 데이터 사이언티스트와 인공지능 연구자들은 모델을 학습시키는 것 보다 데이터 관리와 전처리에 대부분의 시간과 노력을 투자하는데, Hub를 통해 이 문제를 해결할 수 있습니다. 저희는 여러분의 데이터셋들을 petabyte 규모까지도 numpy 형태의 어레이로 클라우드에 저장하여 어떤 컴퓨터에서든 원활하게 접근하고 작업할 수 있게끔 합니다. Hub를 이용하면 이미지, 텍스트, 오디오, 비디오 등의 데이터들을 미리 다운로드하지 않고도 로컬에 저장된 것처럼 빠르게 접근하고, 또 활용할 수 있습니다. 또한 팀원들끼리 하나의 데이터셋을 동기화하여 사용할 수도 있습니다.

Hub 는 Waymo, Red Cross, World Resources Institute, Omdena 등에서 이미 사용되고 있습니다.



### Hub는 이렇습니다!

* 버전 관리가 가능한 대규모의 데이터셋을 저장하고 검색할 수 있습니다.
* Google Docs를 쓸 때처럼 다수의 데이터 사이언티스트들이 서로 간섭받지 않으면서 동일한 데이터셋을 가지고 실시간 협업할 수 있습니다.
* 동시에 여러 기기에서 접속할 수 있습니다.
* Activeloop 뿐만아니라 로컬, 구글클라우드, S3, Azure 등 다양한 플랫폼에 배포할 수 있습니다.  (Activeloop가 default이며, 또한 무료입니다!) 
* Numpy, Dask, Ray, [PyTorch](https://docs.activeloop.ai/en/latest/integrations/pytorch.html), or [TensorFlow](https://docs.activeloop.ai/en/latest/integrations/tensorflow.html) 등의 다양한 머신러닝 프레임워크들과 연동하는 것이 가능합니다.
* 저장되는 어레이의 크기는 자유입니다. 100k by 100k짜리 이미지도 저장 가능합니다!
* 저장되는 어레이들의 shape또한 제각각으로 할 수 있습니다. 
* 특별한 조작 없이도 빠르게 저장된 데이터들을 [Visualize](http://app.activeloop.ai/?utm_source=github&utm_medium=repo&utm_campaign=readme)할 수 있습니다.

 <p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/activeloopai/Hub/master/docs/visualizer%20gif.gif" width="75%"/>
    </br>
app.activeloop.ai (무료 tool)를 이용하여 Hub에 업로드된 데이터를 visualize한 결과.


</p>


## 시작하기
여러분의 개인 데이터나 공용 데이터를 로컬 또는 아무 클라우드에서나 사용할 수 있습니다.

### 빠르게 공용 데이터셋에 접근하기

공용 데이터셋을 로드하려면, 그걸 활용하기 위해 코드를 작성해야 하고, 또 그러기 위해서는 API에 접근하고 사용법을 익히는 데 많은 시간이 소요되는 것이 현실입니다. Hub를 이용하면 단 2줄의 코드만으로 **단 3분 안에 필요로 하시는 데이터셋을 준비하실 수 있습니다.**

```sh
pip3 install hub
```

여러분은 Hub를 이용하여 몇 줄의 간단한 코드만으로 구성된 절차를 거쳐서 공용 데이터셋에 접근하실 수 있습니다. 예를 들어 [MNIST database](https://app.activeloop.ai/dataset/activeloop/mnist/?utm_source=github&utm_medium=repo&utm_campaign=readme)의 첫 1000개의 이미지들을 numpy array 포맷으로 불러오고 싶으시다면 다음과 같은 코드를 작성하시면 됩니다:

```python
from hub import Dataset

mnist = Dataset("activeloop/mnist")  # loading the MNIST data lazily
# saving time with *compute* to retrieve just the necessary data
mnist["image"][0:1000].compute()
```
[app.activeloop.ai](https://app.activeloop.ai/datasets/popular/?utm_source=github&utm_medium=repo&utm_campaign=readme)에 방문하시면 MNIST 말고도 다양한 다른 공용 데이터셋들에 접근하실 수 있는 링크도 얻으실 수 있습니다.

### 모델 학습시키기

데이터를 로드해서 여러분의 모델을 **바로** 학습시킬 수 있습니다. Hub는 Pytorch, Tensorflow와 연동이 가능하고, 데이터 형식 간의 변환 또한 쉽게 가능합니다.   다음은 Pytorch에서 hub를 이용한 예시입니다:

```python
from hub import Dataset
import torch

mnist = Dataset("activeloop/mnist")
# converting MNIST to PyTorch format
mnist = mnist.to_pytorch(lambda x: (x["image"], x["label"]))

train_loader = torch.utils.data.DataLoader(mnist, batch_size=1, num_workers=0)

for image, label in train_loader:
    # Training loop here
```

### 로컬 데이터셋을 생성하기
만약 여러분이 기존 방식처럼 로컬에서 여러분만의 데이터셋을 사용하고 싶으시다면, 다음과 같이 데이터셋을 생성하실 수도 있습니다:

```python
from hub import Dataset, schema
import numpy as np

ds = Dataset(
    "./data/dataset_name",  # file path to the dataset
    shape = (4,),  # follows numpy shape convention
    mode = "w+",  # reading & writing mode
    schema = {  # named blobs of data that may specify types
    # Tensor is a generic structure that can contain any type of data
        "image": schema.Tensor((512, 512), dtype="float"),
        "label": schema.Tensor((512, 512), dtype="float"),
    }
)

# filling the data containers with data (here - zeroes to initialize)
ds["image"][:] = np.zeros((4, 512, 512))
ds["label"][:] = np.zeros((4, 512, 512))
ds.flush()  # executing the creation of the dataset
```

로컬 데이터 뿐만 아니라 `s3://bucket/path`, `gcs://bucket/path` 또는 Azure 경로까지 정해주실 수 있습니다. [이곳](https://docs.activeloop.ai/en/latest/simple.html#data-storage)에서 클라우드에 저장된 데이터를 다루는 법에 대한 추가적인 정보를 얻으실 수 있습니다.

또한, 공개된 공용 데이터셋인데 Hub에서 그 데이터셋을 찾으실 수 없는 경우에는, [저희에게 요청하실 수 있습니다](https://github.com/activeloopai/Hub/issues/new?assignees=&labels=i%3A+enhancement%2C+i%3A+needs+triage&template=feature_request.md&title=[FEATURE]+New+Dataset+Required%3A+%2Adataset_name%2A). 저희가 가능한 한 모두가 사용 가능할 수 있도록 최선을 다하겠습니다.

### 여러분의 데이터셋을 업로드하고 <ins>어디에서든</ins> 접근할 수 있도록 하는 3단계

1. [Activeloop](https://app.activeloop.ai/register/?utm_source=github&utm_medium=repo&utm_campaign=readme)에 무료 계정을 생성하고 로컬에서 인증합니다:
```sh
activeloop register
activeloop login

# alternatively, add username and password as arguments (use on platforms like Kaggle)
activeloop login -u username -p password
```

2. 데이터셋을 생성한 후 이름을 정하고, 당신의 계정에 업로드합니다. 예시는 다음과 같습니다:

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

3. 전 세계 어디에서나 커맨드 라인 입력이 가능한 어떤 기기로든 당신의 데이터셋에 다음과 같이 접근하실 수 있습니다:

```python
from hub import Dataset

ds = Dataset("username/dataset_name")
```


## 공식 문서

대규모 데이터셋을 업로드하거나 다양한 변환을 적용하는 등 더 복잡한 작업을 수행하기를 원하신다면, 저희의 [공식 문서](http://docs.activeloop.ai/?utm_source=github&utm_medium=repo&utm_campaign=readme)를 확인해주시기 바랍니다.

## 튜토리얼
[examples](https://github.com/activeloopai/Hub/tree/master/examples) 경로에는 Hub에 대해 대략적으로 배워볼 수 있는 예제들이 주피터 노트북 형태로 제공되고 있습니다. 아래는 몇몇 유용한 노트들의 리스트입니다.

| 노트  	|   설명	|   	|
|:---	|:---	|---:	|
| [Uploading Images](https://github.com/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%201a%20-%20Uploading%20Images.ipynb) | Hub에 이미지를 업로드하고 저장하는 방법에 관한 소개입니다. |  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%201a%20-%20Uploading%20Images.ipynb) |
| [Uploading Dataframes](https://github.com/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%201b%20-%20Uploading%20Dataframes.ipynb)  	| 데이터프레임을 Hub에 업로드하는 방법에 관한 소개입니다.  	| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%201b%20-%20Uploading%20Dataframes.ipynb)  	|
| [Uploading Audio](https://github.com/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%201c%20-%20Uploading%20Audio.ipynb) | Hub에서 음성 데이터를 다루는 방법에 관한 설명입니다.|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%201c%20-%20Uploading%20Audio.ipynb) |
| [Retrieving Remote Data](https://github.com/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%202%20-%20Retrieving%20Remote%20Data.ipynb) | 데이터를 검색하는 방법에 관한 설명입니다.| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/activeloopai/Hub/blob/master/examples/tutorial/tutorial/Tutorial%202%20-%20Retrieving%20Remote%20Data.ipynb) |
| [Transforming Data](https://github.com/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%203%20-%20Transforming%20Data.ipynb) | Hub를 이용하여 데이터를 변환하는 방법에 관한 간략한 소개입니다.|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%203%20-%20Transforming%20Data.ipynb) |
| [Dynamic Tensors](https://github.com/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%204%20-%20What%20are%20Dynamic%20Tensors.ipynb) | 데이터의 shape이나 size를 다루는 방법에 관한 내용입니다.|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%204%20-%20What%20are%20Dynamic%20Tensors.ipynb) |
| [NLP using Hub](https://github.com/activeloopai/Hub/blob/master/examples/nlp_using_hub.ipynb) | 실제로 CoLA 데이터셋을 이용해 BERT 모델을 Fine Tune하는 방법에 관한 예제입니다.|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/activeloopai/Hub/blob/master/examples/nlp_using_hub.ipynb) |


## 사용 예시
* **위성과 드론 이미지**: [확장성있는 항로를 통해 농사의 효율 높이기](https://activeloop.ai/usecase/intelinair?utm_source=github&utm_medium=repo&utm_campaign=readme), [인도의 경제 복지 지도 만들기](https://towardsdatascience.com/faster-machine-learning-using-hub-by-activeloop-4ffb3420c005), [적십자와 함께 케냐의 메뚜기 문제 해결하기](https://omdena.com/projects/ai-desert-locust/)
* 의료 영상: MRI나 X-Ray와 같은 볼륨 이미지
* **자율주행 자동차**: [Radar, 3D LIDAR, Point Cloud, Semantic Segmentation, Video Objects](https://medium.com/snarkhub/extending-snark-hub-capabilities-to-handle-waymo-open-dataset-4dc7b7d8ab35)
* **상업**: 셀프 계산대 데이터셋
* **미디어**: 영상, 비디오, 오디오 데이터

## 왜 Hub여야 하는가?

Hub와 유사한 기능을 제공하는 데이터셋 관리 라이브러리가 꽤 존재합니다. 실제로 상당수의 사용자들이 PyTorch나 Tensorflow Dataset들을 Hub로 마이그레이션합니다. 다음은 여러분이 Hub로 오시게 되면 겪게 되실 몇 가지 놀라운 차이점들입니다:

* 데이터셋이 chunk의 형태로 제공되기 때문에 그것들을 한꺼번에 미리 다운로드한 후에 사용할 필요 없이 원격으로 스트리밍하실 수 있습니다.
* 데이터의 필요한 부분만이 평가되기 때문에, 여러분은 즉각적으로 데이터를 활용하실 수 있습니다.
* 여러분이 보유하신 메모리보다 큰 규모의 데이터셋을 저장하고 활용하실 수 있습니다.
* 당신의 데이터셋을 다양한 기기들에 걸쳐 다수의 사용자들과 함께 협업하고 버전도 관리할 수 있습니다.
* Visualization Tool과 같이 여러분이 데이터를 쉽게 이해하실 수 있게 돕는 장치들을 갖출 수 있게 됩니다.
* 다수의 학습 라이브러리를 위해서 단 한번만 데이터를 준비하면 됩니다. (예를 들어, 여러분은 PyTorch와 Tensorflow 각각을 이용해서 학습을 하기 위해 동일한 데이터셋을 사용하실 수 있습니다.)


## 커뮤니티

데이터셋 관리 및 전처리에 관한 최신 정보를 얻고 싶으시거나 Activeloop 팀 또는 다른 사용자들로부터 도움을 받고 싶으시다면 저희 [**Slack 커뮤니티**](https://join.slack.com/t/hubdb/shared_invite/zt-ivhsj8sz-GWv9c5FLBDVw8vn~sxRKqQ)에 참여하세요!

<img alt="tweet" src="https://img.shields.io/twitter/follow/activeloopai?label=stay%20in%20the%20Loop&style=social"> on Twitter.

저희 컨트리뷰터 분들께 항상 진심으로 감사드립니다!    

<a href="https://github.com/activeloopai/hub/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=activeloopai/hub" />
</a>

[contributors-img](https://contrib.rocks)를 통해 만든 이미지입니다.

Hub에 기여하고 싶으시다면 [CONTRIBUTING.md](CONTRIBUTING.md) 문서를 참고해주세요.

## 예시
Activeloop의 Hub 포맷은 여러분께서 적은 리소스로 빠른 인퍼런스 속도를 달성할 수 있게끔 해드립니다. 저희는 30개가 넘는 유명한 데이터셋을 미이 보유중입니다. 대표적으로는 다음과 같습니다:

- COCO
- CIFAR-10
- PASCAL VOC
- Cars196
- KITTI
- EuroSAT 
- Caltech-UCSD Birds 200
- Food101

위 리스트와 저희 [visualizer web app](https://app.activeloop.ai/datasets/popular/?utm_source=github&utm_medium=repo&utm_campaign=readme) 링크를 확인하시고 해당 데이터셋들을 여러분의 모델을 학습시키는데 바로 활용해보세요!

## README 뱃지

Hub를 사용하고 계시다면 모두가 알 수 있게 README에 저희 뱃지를 추가해주세요:


[![hub](https://img.shields.io/badge/powered%20by-hub%20-ff5a1f.svg)](https://github.com/activeloopai/Hub)

```
[![hub](https://img.shields.io/badge/powered%20by-hub%20-ff5a1f.svg)](https://github.com/activeloopai/Hub)
```

## 면책 조항

다른 데이터셋 관리 패키지들과 마찬가지로 `Hub`는 공용 데이터셋을 다운로드하고 준비할 수 있게 하는 유틸리티 라이브러리입니다. 당사는 이러한 데이터셋들을 호스팅 또는 배포하는 것이 아니며, 품질 또는 공정성을 보증하거나 각 데이터셋의 라이선스가 있음을 주장하지 않습니다. 데이터셋 사용 권한에 관한 라이선스를 확인하는 것은 사용자의 책임입니다.

만약 당신이 어떤 데이터셋의 소유자이고 데이터셋의 부가 설명, 인용 정보 등을 추가하고 싶거나 또는 본 라이브러리에 해당 데이터셋이 포함되는 것을 원치 않으실 경우 [GitHub issue](https://github.com/activeloopai/Hub/issues/new)를 통해 알려주시길 바랍니다. ML 커뮤니티에 대한 당신의 기여에 깊은 감사의 말씀을 드립니다.


## 감사의 말
이 기술은 Princeton University에서의 경험에서 영감을 받아 시작되었으며, William Silversmith @SeungLab에게 그의 멋진 [cloud-volume](https://github.com/seung-lab/cloud-volume) 툴에 대해 감사를 표하고 싶습니다. 또한 저희는 [Zarr](https://zarr.readthedocs.io/en/stable/)의 헤비 유저이며 그들의 커뮤니티가 이러한 기반을 마련해 준 점에 대해서도 큰 감사의 말씀 드립니다.

