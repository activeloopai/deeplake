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

---

</a>
</p>

<h3 align="center"> Hub 向您展示数据的新纪元 </br> 以最快的方式储存、访问和管理 PyTorch与TensorFlow数据，并建立可扩展的数据工作流 </h3>

---

[ [English](./README.md) | 简体中文 ]

### Hub 的作用是什么?

新时代的的软件需要新时代的数据，而 Hub 提供这些数据。数据科学家与机器学习研究者常常花费大量时间管理与预处理数据，因而牺牲了训练模型的时间。为了改进这一现状，我们创造了 Hub 。我们将您可达PB量级的数据转换为单个类numpy数组，将其存储在云端，使您可以无缝地从任何设备访问您的数据。Hub 使任何储存在云端的数据类型（图像、文本、音频或视频）像在本地服务器一样能被快速使用。通过使用一致的数据库，您的小组可以一直保持同步。

Waymo、红十字会、世界资源协会、Omdena 等组织都在使用 Hub。

### 特点 

* 通过版本控制工具储存和获取大型数据库
* 像 Google Docs 一样协作: 多个数据科学家不间断地同时处理一组数据
* 同时从多个设备访问
* 部署在任何地方 - 本地、Google Cloud、S3、Azure或是Activeloop (默认——并且免费！) 
* 与您的机器学习工具整合， 比如 Numpy、Dask、Ray、[PyTorch](https://docs.activeloop.ai/en/latest/integrations/pytorch.html)或[TensorFlow](https://docs.activeloop.ai/en/latest/integrations/tensorflow.html)
* 随心所欲地创建任意大小的数组. 您甚至可以储存 100k x 100k 大小的图片!
* 样本的形状是动态的. 因此您可以把不同大小的数组储存在一个数组内
* 无需冗长的操作，用几秒种即可[可视化](http://app.activeloop.ai/?utm_source=github&utm_medium=repo&utm_campaign=readme)数据中的片段

<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/activeloopai/Hub/master/docs/visualizer%20gif.gif" width="75%"/>
    </br>
使用app.activeloop.ai（免费工具）可视化上传到Hub的数据集。

</p>

## 如何开始

### 访问公共数据集. 超快

以前为了加载一个公共数据集，大家需要很多代码，浪费很多时间来访问和理解 API ，和下载数据。如果使用 Hub, 只需要2行代码， 您即可**在3分钟内开始工作**。

```sh
pip3 install hub
```

用 Hub 访问公共数据集只需要几行代码的简单步骤。运行这个片段就可以 numpy 数组的形式取得[MNIST 数据库](https://app.activeloop.ai/dataset/activeloop/mnist/?utm_source=github&utm_medium=repo&utm_campaign=readme)前1000张图片。

```python
from hub import Dataset

mnist = Dataset("activeloop/mnist")  # loading the MNIST data lazily
# saving time with *compute* to retrieve just the necessary data
mnist["image"][0:1000].compute()
```
您可以在 [app.activeloop.ai](https://app.activeloop.ai/datasets/popular/?utm_source=github&utm_medium=repo&utm_campaign=readme) 找到其他流行的数据库.

### 训练模型

加载数据并且**直接地**训练您的模型。Hub 已与 PyTorch 和 TensorFlow 整合，并且以通俗的方式进行格式转换。看看下面使用 PyTorch 的例子：

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

### 创建本地数据集 
如果您想在本地处理您的数据，您可以从创建一个数据集开始：

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
ds.commit()  # executing the creation of the dataset
```

您也可以指明 `s3://bucket/path`, `gcs://bucket/path` 或 azure 路径。您可以在[这里](https://docs.activeloop.ai/en/latest/simple.html#data-storage)找到云储存的更多相关信息。

如果您需要 Hub 里目前找不到的公共数据集，您可以发送 [Issues](https://github.com/activeloopai/Hub/issues/new?assignees=&labels=i%3A+enhancement%2C+i%3A+needs+triage&template=feature_request.md&title=[FEATURE]+New+Dataset+Required%3A+%2Adataset_name%2A)。我们将尽快为所有人实现他。

### 用简单的3步上传您的数据集 并 从<ins>任何地方</ins>访问它

1. 在 [Activeloop](https://app.activeloop.ai/register/?utm_source=github&utm_medium=repo&utm_campaign=readme) 注册一个免费账户并在本地认证:
```sh
hub register
hub login
```

2. 然后创建一个数据集，注明它的名字，然后把它上传到您的账户。比如：
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
ds.commit()
```

3. 在任何地点，以任何机器，只要有命令行就可访问它：
```python
from hub import Dataset

ds = Dataset("username/dataset_name")
```


## 官方文档

对于其他更高级的工作流，比如上传较大的数据集，或者应用多重变换，请阅读[文档](http://docs.activeloop.ai/?utm_source=github&utm_medium=repo&utm_campaign=readme)。

## 应用场景
* **卫星和无人机成像**: [可扩展的航空数据流](https://activeloop.ai/usecase/intelinair?utm_source=github&utm_medium=repo&utm_campaign=readme), [绘制印度的经济状况](https://towardsdatascience.com/faster-machine-learning-using-hub-by-activeloop-4ffb3420c005), [与红十字一起在肯尼亚抗击沙漠蝗虫](https://omdena.com/projects/ai-desert-locust/)
* **医学图像**: 体积图像： MRI， Xray
* **自动驾驶汽车**: [雷达, 3D LIDAR, 点云, 语义分割, 图像对象](https://medium.com/snarkhub/extending-snark-hub-capabilities-to-handle-waymo-open-dataset-4dc7b7d8ab35)
* **零售**: 自行结账数据集
* **媒体**: 图像，视频，音频储存

## 社区

加入我们的 [**Slack 社区**](https://join.slack.com/t/hubdb/shared_invite/zt-ivhsj8sz-GWv9c5FLBDVw8vn~sxRKqQ) 以便从 Activeloop 团队和其他用户获取帮助, 并且与最好的技术保持更新！

<img alt="tweet" src="https://img.shields.io/twitter/follow/activeloopai?label=stay%20in%20the%20Loop&style=social"> on Twitter.

一如既往，感谢我们可爱的贡献者们！ </br>

[![](http://sourcerer.io/fame/davidbuniat/activeloopai/Hub/images/0)](http://sourcerer.io/fame/davidbuniat/activeloopai/Hub/links/0)[![](http://sourcerer.io/fame/davidbuniat/activeloopai/Hub/images/1)](http://sourcerer.io/fame/davidbuniat/activeloopai/Hub/links/1)[![](http://sourcerer.io/fame/davidbuniat/activeloopai/Hub/images/2)](http://sourcerer.io/fame/davidbuniat/activeloopai/Hub/links/2)[![](http://sourcerer.io/fame/davidbuniat/activeloopai/Hub/images/3)](http://sourcerer.io/fame/davidbuniat/activeloopai/Hub/links/3)[![](http://sourcerer.io/fame/davidbuniat/activeloopai/Hub/images/4)](http://sourcerer.io/fame/davidbuniat/activeloopai/Hub/links/4)[![](http://sourcerer.io/fame/davidbuniat/activeloopai/Hub/images/5)](http://sourcerer.io/fame/davidbuniat/activeloopai/Hub/links/5)[![](http://sourcerer.io/fame/davidbuniat/activeloopai/Hub/images/6)](http://sourcerer.io/fame/davidbuniat/activeloopai/Hub/links/6)[![](http://sourcerer.io/fame/davidbuniat/activeloopai/Hub/images/7)](http://sourcerer.io/fame/davidbuniat/activeloopai/Hub/links/7)

请阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 以了解如何为 Hub 做贡献。

## 例子
Activeloop 的 Hub 形使您可以以更低的成本来达成更快的推理。我们的平台已有超过30个流行的数据集。包括：
- COCO
- CIFAR-10
- PASCAL VOC
- Cars196
- KITTI
- EuroSAT 
- Caltech-UCSD Birds 200
- Food101

在我们的 [可视化 web app](https://app.activeloop.ai/datasets/popular/?utm_source=github&utm_medium=repo&utm_campaign=readme) 查看这些和很多其他的数据集，然后直接加载它们来进行模型训练。

## README 徽章

在使用 Hub 吗? 添加 README 徽章来让大家知道: 

[![hub](https://img.shields.io/badge/powered%20by-hub%20-ff5a1f.svg)](https://github.com/activeloopai/Hub)

```
[![hub](https://img.shields.io/badge/powered%20by-hub%20-ff5a1f.svg)](https://github.com/activeloopai/Hub)
```

## 免责声明

与其他的数据管理包一样， `Hub` 是一个用于下载和准备公共数据集的应用型库。我们不托管或分发这些数据集，也不保证其质量或是公平性，也不声明您有使用它们的许可。确认您是否有权在数据集的许可下使用该数据集是您的责任。

如果您是数据集所有者，并希望更新数据集的任何部分（描述，引用等），或者不希望将数据集包含在此库中，请通过 [GitHub issue](https://github.com/activeloopai/Hub/issues/new) 告知我们。感谢您对 ML 社区做出的贡献！

## 致敬
这项技术的灵感来自我们在普林斯顿大学的经验，并且感谢 William Silversmith @SeungLab 和他优秀的 [cloud-volume](https://github.com/seung-lab/cloud-volume) 工具。我们是 [Zarr](https://zarr.readthedocs.io/en/stable/) 的重度使用者，特别感谢他们建立了如此重要的基础建设。
