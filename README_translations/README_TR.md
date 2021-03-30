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

<h3 align="center"> Hub ile Data 2.0'a Giriş </br>PyTorch ve Tensorflow için versiyon kontrollü olarak veri seti depolama, erişim ve yönetim işlemleri için en hızlı yol. </h3>

---

[ [English](../README.md) | [Français](./README_FR.md) | [简体中文](./README_CN.md) | Türkçe | [한글](./README_KR.md) | [Bahasa Indonesia](./README_ID.md)]

### Hub ne içindir?

Hub, Yazılım 2.0'ın Data 2.0 gerekliliğini sağlar. Veri bilimciler ve makine öğrenmesi araştırmacıları zamanlarının çoğunu model eğitmek yerine veri yönetimi ve veri önişleme adımlarında harcamaktadır. Hub ile bu sorunu çözüyoruz. Veri setlerinizi (petabayt ölçeğinde olsa bile) bulut üzerinde tek bir parça numpy benzeri dizi olarak depoluyoruz. Böylece bu veri setlerini herhangi bir makineden sorunsuz bir şekilde erişebilir ve üzerinde çalışılabilir hale getiriyoruz. Hub, bulutta depolanan her türlü veri türünü (fotoğraflar, metin dosyaları, ses veya video) on-prem şeklinde depolanmış gibi hızlı bir şekilde kullanılabilmesini sağlar. Aynı veri seti görünümüyle ekibiniz her zaman senkron şekilde çalışabilir.

Hub, Waymo, Kızıl Haç, Dünya Kaynakları Enstitüsü, Omdena ve diğerleri tarafından kullanılmaktadır.

### Özellikler 

* Versiyon kontrol sistemi ile birlikte büyük ölçekli veri setlerinizi depolayın ve erişim erişim sağlayın
* Google Dokümanlardaki gibi birlikte çalışın: Birden fazla veri bilimcinin aynı veri üzerinde çakışma olmadan senkronize bir şekilde çalışabilmesi
* Aynı anda birden fazla cihazdan erişim  
* Herhangi bir yerde ayağa kaldırın(deployment) - lokal, Google Cloud, S3, Azure ve Activeloop (varsayılan olarak ve ücretsiz)   
* Numpy, Dask, Ray, [PyTorch](https://docs.activeloop.ai/en/latest/integrations/pytorch.html), ya da [TensorFlow](https://docs.activeloop.ai/en/latest/integrations/tensorflow.html) gibi ML araçları ile entegre edin
* İstenilen büyüklükte diziler uluşturun. Fotoğraflar, yüz bine yüz bin gibi büyük boyutlarda bile depolanabilir 
* Her veri örneğinin boyutunu (shape) dinamik olarak tutun. Bu sayede küçük ve büyük boyutlu dizileri tek dizi olarak depolayın
* Verilerinizin herhangi bir bölümünü gereksiz manipülasyonlara gerek kalmadan, saniyeler içerisinde [görselleştirin](http://app.activeloop.ai/?utm_source=github&utm_medium=repo&utm_campaign=readme) 

<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/activeloopai/Hub/master/docs/visualizer%20gif.gif" width="75%"/>
    </br>
app.activeloop.ai aracılığıyla Hub'a yüklenen bir veri kümesinin görselleştirilmesi (Ücretsiz Araç).

</p>

## Buradan Başlayın
Yerelde ya da herhangi bir bulut üzerinde, herkese açık ya da kendi verilerinizle çalışın.

### Herkese Açık Verilere Hızlı Erişim Sağlayın

Herkese açık bir veri setini yüklemek için düzinelerce satır kod yazmanıza ve API'yi anlamak ve erişmek için saatlerinizi harcamanız gerekir. Hub ile birlikte tek ihtiyacınız olan 2 satırlık bir kod parçası ve sonrasında **3 dakika içerisinde veri setiniz ile çalışmaya başlayabilirsiniz.**.

```sh
pip3 install hub
```

Basit bir kuralı izleyerek sadece birkaç satırlık kod parçacığı ile Hub'daki herkese açık veri setlerine erişin. [MNIST veri setindeki](https://app.activeloop.ai/dataset/activeloop/mnist/?utm_source=github&utm_medium=repo&utm_campaign=readme) ilk bin fotoğrafı numpy dizisi formatında almak için aşağıdaki kod parçacığını çalıştırabilirsiniz:  
```python
from hub import Dataset

mnist = Dataset("activeloop/mnist")  # MNIST veri setini basitce yukleme
# *compute* ile sadece gerekli verileri alarak zaman kazanın  
mnist["image"][0:1000].compute()
```
Diğer tüm popüler veri setlerine [app.activeloop.ai](https://app.activeloop.ai/datasets/popular/?utm_source=github&utm_medium=repo&utm_campaign=readme) üzerinden ulaşabilirsiniz. 

### Model Eğitimi

Verilerinizi yükleyin ve **direkt** olarak modelinizi eğitin. Hub, PyTorch ve TensorFlow ile entegre edilmiştir ve formatlar arasında arasında basit ve anlaşılır şekilde dönüşümler gerçekleştirebilir. Bunun için aşağıdaki PyTorch örneğini inceleyebilirsiniz:  

```python
from hub import Dataset
import torch

mnist = Dataset("activeloop/mnist")
# MNIST'den PyTorch formatına donusum
mnist = mnist.to_pytorch(lambda x: (x["image"], x["label"]))

train_loader = torch.utils.data.DataLoader(mnist, batch_size=1, num_workers=0)

for image, label in train_loader:
    # Egitim dongusu
```

### Yerel veri seti olusturun   
Eğer kendi verileriniz üzerinde çalışmak isterseniz, veri seti oluşturarak başlayabilirsiniz:   
```python
from hub import Dataset, schema
import numpy as np

ds = Dataset(
    "./data/dataset_name",  # veri setinin dosya yolu
    shape = (4,),  # numpy shape donusumu
    mode = "w+",  # okuma ve yazma modu
    schema = {  # tip tanımlamaları yapılacak isimlendirilmiş veri blokları 
    # Tensor her turlu veriyi icerebilen kapsamlı bir yapidir.
        "image": schema.Tensor((512, 512), dtype="float"),
        "label": schema.Tensor((512, 512), dtype="float"),
    }
)

# veri konteynerlarına veri eklemek (burada sıfırlar ile baslatiliyor)
ds["image"][:] = np.zeros((4, 512, 512))
ds["label"][:] = np.zeros((4, 512, 512))
ds.flush()  # veri setinin derlenmesi
```

Ayrıca `s3://bucket/path`, `gcs://bucket/path` ya da Azure yolu da belirtebilirsiniz. [Buradan](https://docs.activeloop.ai/en/latest/simple.html#data-storage) bulut depolama hakkında daha fazla bilgi edinebilirsiniz. Aynı zamanda Hub'da bulamadığınız herkese açık veri setleri için [dosya talebinde](https://github.com/activeloopai/Hub/issues/new?assignees=&labels=i%3A+enhancement%2C+i%3A+needs+triage&template=feature_request.md&title=[FEATURE]+New+Dataset+Required%3A+%2Adataset_name%2A) bulunabilirsiniz. Mümkün olan en kısa süre içerisinde veri setini herkese açık olarak etkinleştireceğiz!

### Veri setinizi yükleyin ve <ins> her yerden </ins> 3 basit adımda erişin

1. [Activeloop](https://app.activeloop.ai/register/?utm_source=github&utm_medium=repo&utm_campaign=readme)'a ücretsiz bir şekilde üye olun ve yerel olarak giriş yapın:
```sh
activeloop register
activeloop login

# Eger isterseniz parametre olarak kullanıcı adı ve parola ekleyin (Kaggle gibi platformlarda kullanılır) 
activeloop login -u username -p password
```

2. Ardından isim vererek veri setinizi oluşturun ve bunu hesabınıza yükleyin. Örneğin:  
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

3. Komut satırına sahip herhangi bir cihazdan, dünyanın herhangi bir yerinden oluşturduğunuz veri setine erişin:  
```python
from hub import Dataset

ds = Dataset("username/dataset_name")
```


## Dokümantasyon

Büyük veri setlerini karşıya yüklemek veya verilere dönüşümler uygulamak gibi daha gelişmiş pipeline'lar için lütfen [dokümantasyonu](http://docs.activeloop.ai/?utm_source=github&utm_medium=repo&utm_campaign=readme) inceleyin.

## Eğitim Notebookları
Hub'ı genel olarak incelemenizi sağlayacak örnekler için [examples](https://github.com/activeloopai/Hub/tree/master/examples) dizinini inceleyebilirsiniz. Bazı notebooklar aşağıda listelenmiştir.

| Notebook  	|   Açıklama	|   	|
|:---	|:---	|---:	|
| [Fotoğraf Yükleme](https://github.com/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%201a%20-%20Uploading%20Images.ipynb) | Hub'a  fotoğraf yüklemek ve depolamak |  [![Colab'da aç](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%201a%20-%20Uploading%20Images.ipynb) |
| [Dataframe Yüklemek](https://github.com/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%201b%20-%20Uploading%20Dataframes.ipynb)  	| Hub'a DataFrame yüklemek  	| [![Colab'da aç](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%201b%20-%20Uploading%20Dataframes.ipynb)  	|
| [Ses Dosyası Yükleme](https://github.com/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%201c%20-%20Uploading%20Audio.ipynb) | Hub'da ses verilerinin nasıl işlenir|[![Colab'da aç](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%201c%20-%20Uploading%20Audio.ipynb) |
| [Uzak Verileri Almak](https://github.com/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%202%20-%20Retrieving%20Remote%20Data.ipynb) | Uzak verilere erişim nasıl sağlanır| [![Colab'da aç](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/activeloopai/Hub/blob/master/examples/tutorial/tutorial/Tutorial%202%20-%20Retrieving%20Remote%20Data.ipynb) |
| [Veri Dönüştürme](https://github.com/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%203%20-%20Transforming%20Data.ipynb) | Hub ile veri dönüştürme işlemleri ile alakalı bilgiler|[![Colab'da aç](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%203%20-%20Transforming%20Data.ipynb) |
| [Dinamik Tensörler](https://github.com/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%203%20-%20Transforming%20Data.ipynb) | Farklı büyüklükteki ya da farklı shape verileri işleme|[![Colab'da aç](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%203%20-%20Transforming%20Data.ipynb) |
| [Hub Kullanarak NLP](https://github.com/activeloopai/Hub/blob/master/examples/nlp_using_hub.ipynb) | CoLA için Bert modeline ince ayar yapmak|[![Colab'da aç](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/activeloopai/Hub/blob/master/examples/nlp_using_hub.ipynb) |


## Use Cases
* **Uydu ve Drone Görüntüleri**: [Ölçeklenebilir hava pipelineları ile daha akıllı tarım](https://activeloop.ai/usecase/intelinair?utm_source=github&utm_medium=repo&utm_campaign=readme), [Hindistan Refah Haritalanması](https://towardsdatascience.com/faster-machine-learning-using-hub-by-activeloop-4ffb3420c005), [Kenya'da Kızıl Haç ile Çöl Çekirgeleri Arasındaki Mücadele](https://omdena.com/projects/ai-desert-locust/)
* **Tıbbi Görüntüler**: MRI veya Xray gibi hacimsel görüntüler
* **Sürücüsüz Otomobiller**: [Radar, 3D LIDAR, Nokta Bulutu, Semantik Segmentasyon, Video Nesneler](https://medium.com/snarkhub/extending-snark-hub-capabilities-to-handle-waymo-open-dataset-4dc7b7d8ab35)
* **Parekende**: Kendinden kontrollü veri setleri (self-checkout)
* **Medya**: Fotoğraflar, Video, Ses depolaması

## Topluluk

[**Slack Topluluğumuza**](https://join.slack.com/t/hubdb/shared_invite/zt-ivhsj8sz-GWv9c5FLBDVw8vn~sxRKqQ) katılarak Activeloop takımı ve diğer kullanıcılardan yardım alabilirsiniz aynı zamanda veri önişleme ve veri seti yönetimi örnek uygulamaları ile güncel kalabilirsiniz.

Twitter'daki <img alt="tweet" src="https://img.shields.io/twitter/follow/activeloopai?label=gelişmelerden%20haberdar%20olun%20&style=social"> 

Her zaman olduğu gibi, harika katılımcılarımıza teşekkürler!  

<a href="https://github.com/activeloopai/hub/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=activeloopai/hub" />
</a>

[contributors-img](https://contrib.rocks) ile yapılmıştır.

Hub'a nasıl katkıda bulunabileceğinizi öğrenmek için lütfen [CONTRIBUTING.md](CONTRIBUTING.md)'yi okuyunuz.

## Örnekler
Activeloop'un Hub formatı, daha düşük maliyetle daha hızlı çıkarımlar elde etmenizi sağlar. hâlihazırda platformda 30'dan fazla popüler veri seti bulunmaktadır. Bunlar arasında:
- COCO
- CIFAR-10
- PASCAL VOC
- Cars196
- KITTI
- EuroSAT 
- Caltech-UCSD Birds 200
- Food101

vardır. Bu veri setleri ve daha birçok popüler veri seti için [görselleştime web uygulamamıza](https://app.activeloop.ai/datasets/popular/?utm_source=github&utm_medium=repo&utm_campaign=readme) göz atabilirsiniz ve bu veri setlerini model eğitimi için direkt olarak yükleyebilirsiniz.

## README Rozeti

Hub mı kullanıyorsunuz? Herkesin bilmesi için README dosyanıza rozet ekleyebilirsiniz: 


[![hub](https://img.shields.io/badge/powered%20by-hub%20-ff5a1f.svg)](https://github.com/activeloopai/Hub)

```
[![hub](https://img.shields.io/badge/powered%20by-hub%20-ff5a1f.svg)](https://github.com/activeloopai/Hub)
```

## Feragatname

Diğer veri kümesi yönetimi paketlerine benzer şekilde `Hub`, herkese açık veri setlerini indirip hazırlayan bir yardımcı program kütüphanesidir. `Hub` paketi kapsamında bu veri setleri barındırılmamaktadır veya dağıtılmamaktadır, kalitelerine veya yasallığına kefil olunmamaktadır veya kullanıcıların veri kümesini kullanmak için lisansının olduğunu iddia edilmemektedir. Veri setini veri setinin lisansı kapsamında kullanma izninin olup olmadığını belirlemek kullanıcıların sorumluluğundadır.

Bir veri seti sahibiyseniz ve herhangi bir bölümünü (açıklama, alıntı, vb.) güncellemek istiyorsanız veya veri kümenizin bu kütüphaneye dahil edilmesini istemiyorsanız, lütfen bir [GitHub issue](https://github.com/activeloopai/Hub/issues/new) ile tarafımıza bildirin. Makine öğrenimi topluluğuna katkınız için teşekkürler!


## Teşekkürler
 Bu teknolojide Princeton Üniversitesi'ndeki deneyimimizden esinlenilmiştir ve sunmuş olduğu harika [cloud-volume](https://github.com/seung-lab/cloud-volume) aracı için William Silversmith ve @SeungLab'a teşekkür ederiz. [Zarr](https://zarr.readthedocs.io/en/stable/)'ı yoğun olarak kullanıyoruz. Böylesine büyük bir ana blok inşa ettikleri için topluluklarına teşekkür ederiz. 
