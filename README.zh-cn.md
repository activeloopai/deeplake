<img src="https://static.scarf.sh/a.png?x-pxid=bc3c57b0-9a65-49fe-b8ea-f711c4d35b82" /><p align="center">
     <img src="https://user-images.githubusercontent.com/83741606/156426873-c0a77da0-9e0f-41a0-a4fb-cf77eb2fe35e.png" width="300"/>
</h1>
    </br>
    <h1 align="center">人工智能的数据集格式
 </h1>
<p align="center">
    <a href="https://github.com/activeloopai/Hub/actions/workflows/test-pr-on-label.yml"><img src="https://github.com/activeloopai/Hub/actions/workflows/test-push.yml/badge.svg" alt="PyPI version" height="18"></a>
    <a href="https://pypi.org/project/hub/"><img src="https://badge.fury.io/py/hub.svg" alt="PyPI version" height="18"></a>
    <a href="https://pepy.tech/project/hub"><img src="https://static.pepy.tech/personalized-badge/hub?period=month&units=international_system&left_color=grey&right_color=orange&left_text=Downloads" alt="PyPI version" height="18"></a>
     <a href="https://github.com/activeloopai/Hub/issues">
    <img alt="GitHub issues" src="https://img.shields.io/github/issues/activeloopai/Hub"> </a>
    <a href="https://codecov.io/gh/activeloopai/Hub/branch/main"><img src="https://codecov.io/gh/activeloopai/Hub/branch/main/graph/badge.svg" alt="codecov" height="18"></a>
  <h3 align="center">
   <a href="https://docs.activeloop.ai/?utm_source=github&utm_medium=github&utm_campaign=github_readme&utm_id=readme"><b>文档</b></a> &bull;
   <a href="https://docs.activeloop.ai/getting-started/?utm_source=github&utm_medium=github&utm_campaign=github_readme&utm_id=readme"><b>入门</b></a> &bull;
   <a href="https://api-docs.activeloop.ai/"><b>API 参考</b></a> &bull;  
   <a href="https://github.com/activeloopai/examples/"><b>例子</b></a> &bull; 
   <a href="https://www.activeloop.ai/resources/?utm_source=github&utm_medium=github&utm_campaign=github_readme&utm_id=readme"><b>博客</b></a> &bull;  
  <a href="http://slack.activeloop.ai"><b>Slack 社区</b></a> &bull;
  <a href="https://twitter.com/intent/tweet?text=The%20dataset%20format%20for%20AI.%20Stream%20data%20to%20PyTorch%20and%20Tensorflow%20datasets&url=https://activeloop.ai/&via=activeloopai&hashtags=opensource,pytorch,tensorflow,data,datascience,datapipelines,activeloop,databaseforAI"><b>Twitter（推特）</b></a>
 </h3>
 
 
*用其他语言阅读这篇文章: [English](README.md)*

## 关于 Hub

Hub 是一种数据集格式，提供简单的 API 以用于创建、存储和协作处理任何规模的 AI 数据集。Hub 的数据布局可在大规模训练模型的同时实现数据的快速转换和流式传输。谷歌、Waymo、红十字会、牛津大学都在使用 Hub。 Hub 包括以下功能：

<details>
  <summary><b>与存储无关的 API</b></summary>
使用相同的 API 向/从 AWS S3/S3 兼容存储、GCP、Activeloop 云、本地存储以及内存中上传、下载和流式传输数据集。
</details>
<details>
  <summary><b>压缩存储</b></summary>
以原生压缩方式存储图像、音频和视频，仅在需要时解压缩，例如在训练模型时。
</details>
<details>
  <summary><b>惰性 NumPy 类索引</b></summary>
将 S3 或 GCP 数据集视为系统内存中 NumPy 数组的集合。对它们进行切片、索引或迭代。只会下载您要求的字节！
</details>
<details>
  <summary><b>数据集版本控制</b></summary>
提交、创建分支、切换分支 - 您在代码存储库中已经熟悉的概念现在也可以应用于您的数据集！
</details>
<details>
  <summary><b>与深度学习框架的集成</b></summary>
Hub 带有 Pytorch 和 Tensorflow 的内置集成。用几行代码训练你的模型——我们甚至负责数据集洗牌。:)
</details>
<details>
  <summary><b>分布式转换</b></summary>
使用多线程、多处理或我们的内置<a href="https://www.ray.io/">Ray</a>集成快速在您的数据集进行转换操作。
</details>
<details>
  <summary><b>在几秒钟内可用100 多个最流行的图像、视频和音频数据集</b></summary>
Hub 社区已经上传<a href="https://docs.activeloop.ai/datasets/?utm_source=github&utm_medium=github&utm_campaign=github_readme&utm_id=readme">100 多个图像、视频和音频数据集等</a> 例如 <a href="https://docs.activeloop.ai/datasets/mnist/?utm_source=github&utm_medium=github&utm_campaign=github_readme&utm_id=readme">MNIST</a>, <a href="https://docs.activeloop.ai/datasets/coco-dataset/?utm_source=github&utm_medium=github&utm_campaign=github_readme&utm_id=readme">COCO</a>,  <a href="https://docs.activeloop.ai/datasets/imagenet-dataset/?utm_source=github&utm_medium=github&utm_campaign=github_readme&utm_id=readme">ImageNet</a>,  <a href="https://docs.activeloop.ai/datasets/cifar-10-dataset/?utm_source=github&utm_medium=github&utm_campaign=github_readme&utm_id=readme">CIFAR</a>,  <a href="https://docs.activeloop.ai/datasets/gtzan-genre-dataset/?utm_source=github&utm_medium=github&utm_campaign=github_readme&utm_id=readme">GTZAN</a> 等等.
</details>
</details>
<details>
  <summary><b><a href="https://app.activeloop.ai/?utm_source=github&utm_medium=github&utm_campaign=github_readme&utm_id=readme">Activeloop平台</a>提供即时可视化支持</b></summary>
Hub 数据集在<a href="https://app.activeloop.ai/?utm_source=github&utm_medium=github&utm_campaign=github_readme&utm_id=readme">Activeloop 平台</a> 中通过边界框、掩码、注释等立即实现可视化（见下文）.
</details>


<div align="center">
<a href="https://www.linkpicture.com/view.php?img=LPic61b13e5c1c539681810493"><img src="https://www.linkpicture.com/q/ReadMe.gif" type="image"></a>
</div>

    
## Hub 入门


### 🚀 如何安装 Hub
Hub 是用 100% Python 编写的，可以使用 pip 快速安装。

```sh
pip3 install hub
```

**默认情况下，Hub 不安装音频、视频和谷歌云(GCS)支持的依赖项。它们可以使用以下方式安装：**
```sh
pip3 install hub[video]  -> 通过 pyav 支持视频
pip3 install hub[gcp]    -> 通过 google-* 依赖支持GCS
pip3 install hub[all]    -> 安装所有东西 - 音频、视频和GCS支持
```

### 🧠 在 Hub 数据集上训练 PyTorch 模型

#### 加载 CIFAR 10，这是 Hub 中现成的数据集之一：

```python
import hub
import torch
from torchvision import transforms, models

ds = hub.load('hub://activeloop/cifar10-train')
```

#### 检查数据集中的张量：

```python
ds.tensors.keys()    # dict_keys(['images', 'labels'])
ds.labels[0].numpy() # array([6], dtype=uint32)
```

#### 在 CIFAR 10 数据集上训练 PyTorch 模型，无需下载
首先，为图像定义一个转换并使用 Hub 的内置 PyTorch 单行数据加载器将数据连接到计算：

```python
tform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

hub_loader = ds.pytorch(num_workers=0, batch_size=4, transform={
                        'images': tform, 'labels': None}, shuffle=True)
```

接下来，定义模型、损失和优化器：

```python
net = models.resnet18(pretrained=False)
net.fc = torch.nn.Linear(net.fc.in_features, len(ds.labels.info.class_names))
    
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```

最后，2 个 epoch 的训练循环：

```python
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(hub_loader):
        images, labels = data['images'], data['labels']
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(images)
        loss = criterion(outputs, labels.reshape(-1))
        loss.backward()
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
```


### 🏗️ 如何创建一个 Hub 数据集

可以在不同地址（存储提供商）创建 hub 数据集。这就是他们每个路径：

| 存储提供商       | 示例路径                       |
|------------------|--------------------------------|
| Activeloop 云    | hub://user_name/dataset_name   |
| AWS S3 / S3 兼容 | s3://bucket_name/dataset_name  |
| GCP              | gcp://bucket_name/dataset_name |
| 本地存储         | path to local directory        |
| 内存             | mem://dataset_name             |



让我们在 Activeloop 云中创建一个数据集。Activeloop 云为每位用户提供高达 300 GB 的免费存储空间（更多信息请点击[此处](#-对于学生和教育工作者)）。如果您还没有账号，请从终端使用`activeloop register` 创建一个新帐户。系统将要求您输入用户名、电子邮件 ID 和密码。您在此处输入的用户名将用于数据集路径。

```sh
$ activeloop register
Enter your details. Your password must be at least 6 characters long.
Username:
Email:
Password:
```

在 Activeloop Cloud 中初始化一个空数据集：

```python
import hub

ds = hub.empty("hub://<USERNAME>/test-dataset")
```


接下来，创建一个张量来保存我们刚刚初始化的数据集中的图像：

```python
images = ds.create_tensor("images", htype="image", sample_compression="jpg")
```

假设您有一个图像文件路径列表，让我们将它们上传到数据集：

```python
image_paths = ...
with ds:
    for image_path in image_paths:
        image = hub.read(image_path)
        ds.images.append(image)
```

或者，您也可以上传 numpy 数组。由于 `images` 张量是使用 `sample_compression="jpg"` 创建的，数组将使用 jpeg 压缩进行压缩。

```python
import numpy as np

with ds:
    for _ in range(1000):  # 1000 random images
        random_image = np.random.randint(0, 256, (100, 100, 3))  # 100x100 image with 3 channels
        ds.images.append(random_image)
```



### 🚀 如何加载 Hub 数据集


您可以使用一行代码加载刚刚创建的数据集：

```python
import hub

ds = hub.load("hub://<USERNAME>/test-dataset")
```

您还可以访问 Hub 格式的<a href="https://docs.activeloop.ai/datasets/?utm_source=github&utm_medium=github&utm_campaign=github_readme&utm_id=readme"> 100 多个图像、视频和音频数据集</a>，而不仅仅是您创建的那些。以下是加载 [Objectron Bikes 数据集](https://github.com/google-research-datasets/Objectron)的方法:

```python
import hub

ds = hub.load('hub://activeloop/objectron_bike_train')
```

要以 numpy 格式获取 Objectron Bikes 数据集中的第一张图像：


```python
image_arr = ds.image[0].numpy()
```



## 📚 文档
可以在我们的[文档页面](http://docs.activeloop.ai/?utm_source=github&utm_medium=repo&utm_campaign=readme)上找到入门指南、示例、教程、API 参考和其他有用信息。

## 🎓 对于学生和教育工作者
Hub 用户可以通过与 Activeloop 平台的免费集成来访问和可视化各种流行的数据集。用户还可以创建和存储自己的数据集，并将其提供给公众。高达 300 GB 的免费存储空间可供学生和教育工作者使用：

| <!-- -->                          | <!-- -->   |
|-----------------------------------|------------|
| Activeloop 托管的公共数据集的存储 | 200GB 免费 |
| Activeloop 托管的私有数据集的存储 | 100GB 免费 |



## 👩‍💻 与熟悉工具的比较


<details>
  <summary><b>Activeloop Hub vs DVC</b></summary>
  
Hub 和 DVC 为数据提供类似于 git 的数据集版本控制，但它们存储数据的方法有很大不同。Hub 将数据转换并存储为分块压缩数组，从而可以快速流式传输到 ML 模型，而 DVC 在存储在效率较低的传统文件结构中的数据之上运行。当数据集由许多文件（即许多图像）组成时，与 DVC 的传统文件结构相比，Hub 格式使数据集版本控制更加容易。另一个区别是 DVC 主要使用命令行界面，而 Hub 是 Python 包。最后，Hub 提供了一个 API，可以轻松地将数据集连接到 ML 框架和其他常见的 ML 工具，并通过[Activeloop 的可视化工具](http://app.activeloop.ai/?utm_source=github&utm_medium=repo&utm_campaign=readme)实现即时数据集可视化.
</details>


<details>
  <summary><b>Activeloop Hub vs TensorFlow 数据集 (TFDS)</b></summary>
  
Hub 和 TFDS 将流行的数据集无缝连接到 ML 框架。Hub 数据集与 PyTorch 和 TensorFlow 兼容，而 TFDS 仅与 TensorFlow 兼容。Hub 和 TFDS 之间的一个关键区别在于，Hub 数据集是为从云端流式传输而设计的，而 TFDS 必须在使用前在本地下载。因此使用 Hub 可以直接从 TensorFlow 数据集导入数据集，并将它们流式传输到 PyTorch 或 TensorFlow。除了提供对流行的公开数据集的访问之外，Hub 还提供强大的工具来创建自定义数据集，将它们存储在各种云存储提供商上，并通过简单的 API 与他人协作。TFDS 主要专注于让公众轻松访问常用数据集，而自定义数据集的管理不是主要关注点。一个详细的对比介绍可以在[这里](https://www.activeloop.ai/resources/tensor-flow-tf-data-activeloop-hub-how-to-implement-your-tensor-flow-data-pipelines-with-hub/)看到.

</details>



<details>
  <summary><b>Activeloop Hub vs HuggingFace</b></summary>
Hub 和 HuggingFace 都提供对流行数据集的访问，但 Hub 主要专注于计算机视觉，而 HuggingFace 专注于自然语言处理。HuggingFace 变换和其他 NLP 计算工具与 Hub 提供的功能不同。

</details>

<details>
  <summary><b>Activeloop Hub vs WebDatasets</b></summary>
Hub 和 WebDatasets 都提供跨网络的快速数据流。它们具有几乎相同的传输速度，因为底层网络请求和数据结构非常相似。但是，Hub 提供了卓越的随机访问和改组，其简单的 API 是在 python 中而不是命令行中，并且 Hub 可以对数据集进行简单的索引和修改，而无需重新创建它。

</details>


## 社区

加入我们的 [**Slack 社区**](https://join.slack.com/t/hubdb/shared_invite/zt-ivhsj8sz-GWv9c5FLBDVw8vn~sxRKqQ)，了解有关使用 Hub 进行非结构化数据集管理的更多信息，并从 Activeloop 团队和其他用户那里获得帮助。

通过完成我们的 3 分钟[**调查**](https://forms.gle/rLi4w33dow6CSMcm9)，我们希望得到您的反馈。

一如既往，感谢我们出色的贡献者！ 

<a href="https://github.com/activeloopai/hub/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=activeloopai/hub" />
</a>

使用[contributors-img](https://contrib.rocks)制作.

请阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 开始为 Hub 做出贡献


## 自述文件徽章

使用Hub？添加一个自述文件徽章让大家知道：


[![hub](https://img.shields.io/badge/powered%20by-hub%20-ff5a1f.svg)](https://github.com/activeloopai/Hub)

```
[![hub](https://img.shields.io/badge/powered%20by-hub%20-ff5a1f.svg)](https://github.com/activeloopai/Hub)
```



## 免责声明

<details>
  <summary><b> 数据集许可证</b></summary>

Hub 用户可以访问各种公开可用的数据集。我们不托管或分发这些数据集，不保证它们的质量或公平性，或声称您拥有使用这些数据集的许可。您有责任确定您是否有权使用其许可下的数据集。

如果您是数据集所有者并且不希望您的数据集包含在此库中，请通过[GitHub issue](https://github.com/activeloopai/Hub/issues/new)与我们联系。感谢您对 ML 社区的贡献！

</details>

<details>
  <summary><b> 使用情况跟踪</b></summary>

默认情况下，我们使用 Bugout 收集使用数据（这是执行此操作的[code](https://github.com/activeloopai/Hub/blob/853456a314b4fb5623c936c825601097b0685119/hub/__init__.py#L24)). 它不收集匿名 IP 地址数据以外的用户数据，并且只记录 Hub 库自己的操作。这有助于我们的团队了解该工具的使用方式以及如何构建对您而言重要的功能！在您注册 Activeloop 后，数据不再是匿名的。您始终可以使用以下 CLI 命令选择退出报告：

```
activeloop reporting --off
```
</details>

## 致谢
这项技术的灵感来自我们在普林斯顿大学的研究工作。我们要感谢 William Silversmith @SeungLab 提供出色的[cloud-volume](https://github.com/seung-lab/cloud-volume) 工具.
