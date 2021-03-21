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

<h3 align="center"> Memperkenalkan <i>Data 2.0</i>, didukung oleh Hub. </br>Cara tercepat untuk mengakses dan manajemen dataset dengan <i>version-control</i> untuk PyTorch/Tensorflow. Dapat bekerja secara lokal atau dalam <i>cloud</i> apa pun. <i>Pipeline</i> data yang dapat diskalakan.</h3>

---

[ [English](../README.md) | [Français](./README_FR.md) | [简体中文](./README_CN.md) | [Türkçe](./README_TR.md) | [한글](./README_KR.md) | Bahasa Indonesia ]

<i>Perhatian: translasi ini mungkin bukan berasal dari dokumen yang paling baru</i>

### Untuk apa Hub digunakan?

Software 2.0 membutuhkan Data 2.0, dan Hub menyediakan ini. Lebih banyak waktu yang dihabiskan oleh <i>Data Scientists</i> atau <i>ML researchers</i> untuk mengatur data dan <i>preprocessing</i> data daripada melakukan <i>training</i> model. Dengan Hub, kami akan memperbaiki ini. Kami menyimpan dataset milikmu (hingga berskala <i>petabyte</i>) sebagai sebuah <i>array</i> tunggal serupa <i>numpy</i> di <i>cloud</i> sehingga kamu dapat dengan mudah mengaksesnya dan menggunakannya dari perangkat manapun. Hub membantu tipe data apapun (gambar, file teks, audio, atau video) yang disimpan dalam <i>cloud</i> dapat digunakan secara cepat seakan-akan data tersebut disimpan dalam penyimpanan lokal. Dengan pandangan data yang sama, tim kalian dapat selalu tersonkrinisasi. 

Hub juga digunakan oleh Waymo, Red Cross, World Resources Institute, Omdena, dan lainnya.

### Fitur-fitur

* Simpan dan ambil dataset besar menggunakan *version-control*
* Berkolaborasi seakan menggunakan Google Docs: Beberapa <i>data scientist</i> mengerjakan data yang sama secara sinkron tanpa gangguan
* Dapat diakses dari beberapa perangkat secara bersamaan
* Gunakan dimana saja - penyimpanan lokal, Google Cloud, S3, Azure atau juga di Activeloop (pengaturan tetap - tanpa biaya!)
* Integrasikan dengan <i>tool</i> ML seperti Numpy, Dask, Ray, [PyTorch](https://docs.activeloop.ai/en/latest/integrations/pytorch.html), atau [TensorFlow](https://docs.activeloop.ai/en/latest/integrations/tensorflow.html)
* Buatlah <i>array</i> sebesar yang kamu mau. Kamu dapat menyimpan gambar-gambar yang sangat besar seperti ukuran 100k x 100k
* Jaga ukuran tiap sampel menjadi dinamis. Dengan cara ini, kamu dapat menyimpan <i>array</i> kecil dan <i>array</i> besar dalam 1 buah <i>array</i> yang sama
* [Visualisasikan](http://app.activeloop.ai/?utm_source=github&utm_medium=repo&utm_campaign=readme) tiap potongan data dalam hitungan detik tanpa manipulasi yang berlebihan.

 <p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/activeloopai/Hub/master/docs/visualizer%20gif.gif" width="75%"/>
    </br>
    
Visualisasi sebuah dataset yang diunggah ke Hub melalui app.activeloop.ai (aplikasi gratis).

</p>


## Panduan Memulai
Bekerja dengan data terbuka atau data milik sendiri, pada peyimpanan lokal atau di penyimpanan <i>cloud</i> mana pun.

### Akses data terbuka. Cepat.

Untuk memuat data terbuka, kamu harus menulis banyak baris kode dan menghabiskan berjam-jam untuk mengakses dan memahami API yang digunakan serta mengunduh datanya. Dengan Hub, kamu hanya membutuhkan 2 baris kode dan kamu dapat **mulai mengolah dataset dalam waktu kurang dari 3 menit**

```sh
pip3 install hub
```

Akses dataset terbuka (<i>public</i>) dengan mudah yang tersedia di Hub menggunakan beberapa baris kode. Jalankan kutipan kode berikut untuk dapat mengakses 1000 data pertama dari [MNIST database](https://app.activeloop.ai/dataset/activeloop/mnist/?utm_source=github&utm_medium=repo&utm_campaign=readme) dalam format <i>numpy array</i>:

```python
from hub import Dataset

mnist = Dataset("activeloop/mnist")  # memuat data MNIST dengan mode lazy
# menghemat waktu dengan *compute* untuk mengambil data yang diperlukan saja
mnist["image"][0:1000].compute()
```

Kamu dapat mencari dataset populer lainnya di [app.activeloop.ai](https://app.activeloop.ai/datasets/popular/?utm_source=github&utm_medium=repo&utm_campaign=readme).

### *Train* sebuah model

Muat datanya dan <i>train</i> modelmu secara *langsung*. Hub sudah terintegrasi dengan PyTorch dan Tensorflow yang mampu melakukan perubahan format data dengan cara yang mudah dipahami. Lihatlah contoh dibawah ini untuk menggunakannya dengan PyTorch:

```python
from hub import Dataset
import torch

mnist = Dataset("activeloop/mnist")
# mengonversi MNIST menjadi format PyTorch
mnist = mnist.to_pytorch(lambda x: (x["image"], x["label"]))

train_loader = torch.utils.data.DataLoader(mnist, batch_size=1, num_workers=0)

for image, label in train_loader:
    # Perulangan proses training disini
```

### Buat sebuah dataset lokal 
Jika kamu ingin bekerja menggunakan datamu sendiri secara lokal, kamu dapat mulai membuat datasetmu dengan cara ini:

```python
from hub import Dataset, schema
import numpy as np

ds = Dataset(
    "./data/dataset_name",  # lokasi penyimpanan data
    shape = (4,),  # mengikuti penulisan format shape pada numpy
    mode = "w+",  # mode menulis & membaca
    schema = {  # penyimpanan bernama yang memiliki tipe tertentu
    # Tensor adalah struktur umum yang dapat berisi bentuk data apa pun
        "image": schema.Tensor((512, 512), dtype="float"),
        "label": schema.Tensor((512, 512), dtype="float"),
    }
)

# mengisi tempat penyimpan data dengan sebuah data 
# (disini - inisialisasi dengan 0)
ds["image"][:] = np.zeros((4, 512, 512))
ds["label"][:] = np.zeros((4, 512, 512))
ds.flush()  # menjalankan proses pembuatan dataset
```

Kamu juga dapat menggunakan `s3://bucket/path`, `gcs://bucket/path` atau azure untuk meyimpannya di <i>cloud</i>. Kamu dapat melihat informasi lebih jelasnya [disini](https://docs.activeloop.ai/en/latest/simple.html#data-storage). Selain itu, jika kamu membutuhkan dataset terbuka yang tidak tersedia di Hub, kamu bisa [mengajukan permintaan](https://github.com/activeloopai/Hub/issues/new?assignees=&labels=i%3A+enhancement%2C+i%3A+needs+triage&template=feature_request.md&title=[FEATURE]+New+Dataset+Required%3A+%2Adataset_name%2A). Kami akan menyediakannya untuk semua orang secepat mungkin! 

### Unggah datasetmu dan akses <ins>di mana pun</ins> dalam 3 langkah mudah

1. Buatlah akun di [Activeloop](https://app.activeloop.ai/register/?utm_source=github&utm_medium=repo&utm_campaign=readme) dan ontentikasi secara lokal:

```sh
activeloop register
activeloop login

# cara lainnya, tambahkan username dan password sebagai argumen
# (seperti yang digunakan pada platform Kaggle)
activeloop login -u username -p password
```

2. Buatlah dataset, dengan merincikan nama dan unggahlah ke akunmu. Contohnya:
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
3. Dapat diakses di mana pun dan perangkat mana pun dengan perintah:
```python
from hub import Dataset

ds = Dataset("username/dataset_name")
```


## Dokumentasi

Untuk <i>pipeline</i> data yang lebih kompleks seperti mengunggah dataset besar atau mengaplikasikan banyak transformasi, silahkan baca [dokumentasi](http://docs.activeloop.ai/?utm_source=github&utm_medium=repo&utm_campaign=readme) kami.

## Tutorial Notebooks

Terdapat berbagai contoh penggunan Hub di direktori [examples](https://github.com/activeloopai/Hub/tree/master/examples) dan beberapa <i>notebook</i> dengan contoh kasus penggunaan juga tersedia di direktori [notebooks](https://github.com/activeloopai/Hub/tree/master/examples/notebooks). Beberapa <i>notebook</i> terdapat dalam list berikut:

| Notebook  	|   Deskripsi	|   	|
|:---	|:---	|---:	|
| [Mengunggah Gambar](https://github.com/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%201a%20-%20Uploading%20Images.ipynb) | Contoh cara mengunggah dan menyimpan Gambar di Hub |  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%201a%20-%20Uploading%20Images.ipynb) |
| [Mengunggah Dataframes](https://github.com/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%201b%20-%20Uploading%20Dataframes.ipynb)  	| Contoh cara mengunggah dan menyimpan <i>Dataframes</i> di Hub  	| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%201b%20-%20Uploading%20Dataframes.ipynb)  	|
| [Uploading Audio](https://github.com/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%201c%20-%20Uploading%20Audio.ipynb) | Menjelaskan cara mengolah data audio in Hub|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%201c%20-%20Uploading%20Audio.ipynb) |
| [Mengambil Data dari penyimpanan <i>cloud</i>](https://github.com/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%202%20-%20Retrieving%20Remote%20Data.ipynb) | Menjelaskan cara mengambil Data dari penyimpanan <i>cloud</i>| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/activeloopai/Hub/blob/master/examples/tutorial/tutorial/Tutorial%202%20-%20Retrieving%20Remote%20Data.ipynb) |
| [Transformasi Data](https://github.com/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%203%20-%20Transforming%20Data.ipynb) | Penjelasan singkat mengenai transformasi data menggunakan Hub|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%203%20-%20Transforming%20Data.ipynb) |
| [Dynamic Tensors](https://github.com/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%204%20-%20What%20are%20Dynamic%20Tensors.ipynb) | Menggunakan data dengan bentuk dan ukuran yang tidak tetap|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%204%20-%20What%20are%20Dynamic%20Tensors.ipynb) |
| [NLP menggunakan Hub](https://github.com/activeloopai/Hub/blob/master/examples/notebooks/nlp_using_hub.ipynb) |  <i>Fine Tuning</i> BERT menggunakan CoLA|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/activeloopai/Hub/blob/master/examples/notebooks/nlp_using_hub.ipynb) |


## Kasus penggunaan
* **Pencitraan satelit and *drone***: [Smarter farming with scalable aerial pipelines](https://activeloop.ai/usecase/intelinair?utm_source=github&utm_medium=repo&utm_campaign=readme), [Mapping Economic Well-being in India](https://towardsdatascience.com/faster-machine-learning-using-hub-by-activeloop-4ffb3420c005), [Fighting desert Locust in Kenya with Red Cross](https://omdena.com/projects/ai-desert-locust/)
* **Gambar Medis**: <i>Volumetric images</i> seperti MRI atau Xray
* **Self-Driving Cars**: [Radar, 3D LIDAR, Point Cloud, Semantic Segmentation, Video Objects](https://medium.com/snarkhub/extending-snark-hub-capabilities-to-handle-waymo-open-dataset-4dc7b7d8ab35)
* **Retail**: <i>Self-checkout</i>  dataset
* **Media**: Gambar, Video, Audio

## Mengapa Hub secara khusus?

Tidak terlalu banyak <i>library</i> manajemen data yang menawarkan fungsionalitas yang mungkin terlihat serupa dengan Hub. Faktanya, cukup banyak pengguna berpindah dari format Dataset PyTorch atau Tensorflow ke Hub. Berikut beberapa perbedaan yang akan kamu temukan setelah beralih ke Hub:
* Data disediakan dalam potongan-potongan (<i>chunks</i>) daripada mengunduhnya dalam satu bagian bersamaan. Hal ini memungkinkan kamu mengunduhnya dari lokasi terpecil
* Hanya bagian yang diperlukan dari dataset yang akan dievaluasi/diolah sehingga kamu dapat segera mengolah datanya
* Kamu dapat menyimpan data yang mungkin tidak dapat kamu simpan secara utuh di memorimu
* kamu dapat mengatur versi datasetmu (<i>version-control</i>) dan berkolaborasi dengan beberapa pengguna dari perangkat yang berbeda-beda
* Dilengkapi dengan <i>tools</i> yang membantumu memahami data dalam hitungan detik, seperti <i>tool</i> visualisasi kami.
* Kamu dapat dengan mudah menyiapkan data untuk digunakan pada proses <i>training</i> oleh beberapa *library* secara bersamaan (contoh, menggunakan satu dataset untuk <i>training</i> dengan PyTorch dan Tensorflow)

## Komunitas

Bergabung dengan kami di [**Komunitas Slack**](https://join.slack.com/t/hubdb/shared_invite/zt-ivhsj8sz-GWv9c5FLBDVw8vn~sxRKqQ) untuk mendapatkan bantuan dari tim Activeloop dan pengguna lainnya serta mengikuti perkembangan terikini dari cara terbaik untuk manajemen data / <i>preprocessing</i> data.

<img alt="tweet" src="https://img.shields.io/twitter/follow/activeloopai?label=stay%20in%20the%20Loop&style=social"> on Twitter.

Seperti biasa, terima kasih kepada kontributor hebat kami!    

<a href="https://github.com/activeloopai/hub/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=activeloopai/hub" />
</a>

Dibuat menggunakan [contributors-img](https://contrib.rocks).

Silahkan baca [CONTRIBUTING.md](CONTRIBUTING.md) untuk mengetahui cara berkontribusi di Hub.

## Kumpulan Dataset
Format Hub yang dibuat Activeloop membantumu melakukan <i>inference</i> lebih cepat dengan <i>cost</i> yang lebih rendah. Kami memiliki 30+ dataset populer tersedia pada platform kami. Beberapa diantaranya:
- COCO
- CIFAR-10
- PASCAL VOC
- Cars196
- KITTI
- EuroSAT 
- Caltech-UCSD Birds 200
- Food101

Periksa dataset diatas dan dataset populer lainnya pada [visualisasi web](https://app.activeloop.ai/datasets/popular/?utm_source=github&utm_medium=repo&utm_campaign=readme) kami dan muat datanya secara langsung untuk <i>training</i> modelmu!

## <i>README Badge</i>

Menggunakan Hub pada pekerjaanmu? Tambahkan <i>README badge</i> agar orang lain mengetahuinya: 


[![hub](https://img.shields.io/badge/powered%20by-hub%20-ff5a1f.svg)](https://github.com/activeloopai/Hub)

```
[![hub](https://img.shields.io/badge/powered%20by-hub%20-ff5a1f.svg)](https://github.com/activeloopai/Hub)
```

## <i>Disclaimers</i>

Serupa dengan manajemen dataset lainnya, `Hub` merupakan <i>utility library</i> untuk mengunduh dan menyiapkan dataset terbuka. Kami tidak menyediakan atau mendistirbusikan dataset-dataset ini, menjamin kualitas atau keadilan dataset tersebut, atau mengklaim Anda memiliki lisensi untuk menggunakan dataset tersebut. Ini merupakan tanggung jawab Anda untuk menentukan apakah Anda memiliki izin untuk menggunakan dataset tersebut sesuai dengan lisensi yang berlaku.

Jika Anda adalah pemilik dari dataset dan ingin memperbarui bagian mana pun (deskripsi, sitasi, dll), atau menginginkan agar dataset tersebut tidak dimasukkan dalam <i>library</i> ini, silahkan hubungi melalui [GitHub issue](https://github.com/activeloopai/Hub/issues/new). Terima kasih atas kontribusi Anda di komunitas ML!

> Similarly to other dataset management packages, `Hub` is a utility library that downloads and prepares public datasets. We do not host or distribute these datasets, vouch for their quality or fairness, or claim that you have license to use the dataset. It is your responsibility to determine whether you have permission to use the dataset under the dataset's license.
> 
> If you're a dataset owner and wish to update any part of it (description, citation, etc.), or do not want your dataset to be included in this library, please get in touch through a [GitHub issue](https://github.com/activeloopai/Hub/issues/new). Thanks for your contribution to the ML community!


## <i>Acknowledgement</i>
Teknologi ini terinspirasi dari pengalaman kami di Princeton University dan ingin berterima kasih kepada William Silversmith @SeungLab dengan tool kerennya [cloud-volume](https://github.com/seung-lab/cloud-volume). Kami adalah pengguna berat [Zarr](https://zarr.readthedocs.io/en/stable/) dan ingin berterima kasih kepada komunitasnya yang telah membangun block fundamental yang hebat.

> This technology was inspired from our experience at Princeton University and would like to thank William Silversmith @SeungLab with his awesome [cloud-volume](https://github.com/seung-lab/cloud-volume) tool. We are heavy users of [Zarr](https://zarr.readthedocs.io/en/stable/) and would like to thank their community for building such a great fundamental block. 
