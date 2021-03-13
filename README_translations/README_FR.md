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

<h3 align="center">  Présentation de Data 2.0, développé par Hub. </br>Le moyen le plus rapide pour stocker, accéder et gérer des ensembles de données avec contrôle de version pour PyTorch/TensorFlow.  Fonctionne localement ou sur n'importe quel Cloud. Pipeline de données évolutif.</h3>

---

[ [English](../README.md) | Français | [简体中文](./README_CN.md) | [Türkçe](./README_TR.md) | [한글](./README_KR.md) | [Bahasa Indonesia](./README_ID.md)]

### À quoi sert Hub ?

Data 2.0 dont a besoin Software 2.0 est fournit par Hub. La plupart du temps, les Data Scientists/ML travaillent sur la gestion et le préparation des données plutôt que sur le training des modèles eux-mêmes. Avec Hub, nous remédions à cette situation. Nous stockons vos ensembles de données (même à l'échelle du pétaoctet) sous la forme d'un tableau numérique unique sur le cloud, de sorte que vous pouvez y accéder et travailler de manière transparente depuis n'importe quelle machine. Hub rend tout type de données (images, fichiers texte, audio ou vidéo) stockées dans le Cloud utilisables aussi rapidement que si elles étaient stockées sur place. Avec la même vue de l'ensemble des données, votre équipe peut toujours être synchronisée. 

Hub est utilisé par Waymo, la Croix-Rouge, le World Resources Institute, Omdena, et d'autres.

### Caractéristiques 

* Stocker et récupérer de grands ensembles de données avec contrôle de version
* Collaborate as in Google Docs: Multiple data scientists working on the same data in sync with no interruptions
* Collaborer comme dans Google Docs : Plusieurs scientifiques travaillant sur les mêmes données en synchronisation sans interruption
* Déployer n'importe où - localement, sur Google Cloud, S3, Azure ainsi qu'Activeloop (par défaut - et gratuitement !) 
* Intégrez avec vos outils de ML comme Numpy, Dask, Ray, [PyTorch](https://docs.activeloop.ai/en/latest/integrations/pytorch.html), ou [TensorFlow](https://docs.activeloop.ai/en/latest/integrations/tensorflow.html)
* Créez des tableaux aussi grands que vous le souhaitez. Vous pouvez stocker des images aussi grandes que 100k par 100k !
* Maintient la forme de chaque échantillon dynamique. De cette façon, vous pouvez stocker les petits et grands tableaux comme un seul tableau
* [Visualiser](http://app.activeloop.ai/?utm_source=github&utm_medium=repo&utm_campaign=readme) toute séquence de données en quelques secondes sans manipulations redondantes

 <p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/activeloopai/Hub/master/docs/visualizer%20gif.gif" width="75%"/>
    </br>
Visualisation d'un ensemble de données téléchargées vers le Hub via app.activeloop.ai (outil gratuit).

</p>


## Pour commencer
Travaillez avec des données publiques ou vos propres données, localement ou n'importe quel Cloud.

### Accéder aux données publiques. Rapidement.

Pour charger un ensemble de données public, il faut écrire des dizaines de lignes de code et passer des heures à accéder à l'API et à la comprendre, ainsi qu'à télécharger les données. Avec Hub, vous n'avez besoin que de deux lignes de code et vous pouvez commencer à travailler sur votre ensemble de données en moins de trois minutes**.

```sh
pip3 install hub
```

Accédez aux ensembles de données publiques dans Hub en suivant une convention simple qui ne nécessite que quelques lignes de code simple. Lancez cet exemple pour obtenir les mille premières images de la [base de données du MNIST] (https://app.activeloop.ai/dataset/activeloop/mnist/?utm_source=github&utm_medium=repo&utm_campaign=readme) au format numpy array :

```python
from hub import Dataset

mnist = Dataset("activeloop/mnist")  # charger les données du MNIST facilement
# gagner du temps avec *compute* pour ne récupérer que les données nécessaires
mnist["image"][0:1000].compute()
```
Vous pouvez trouver tous les autres ensembles de données populaires sur [app.activeloop.ai](https://app.activeloop.ai/datasets/popular/?utm_source=github&utm_medium=repo&utm_campaign=readme).

### Training d'un modèle

Chargez les données et entraînez votre modèle **directement**. Hub est intégré à PyTorch et TensorFlow et effectue des conversions entre formats de manière compréhensible. Regardez l'exemple avec PyTorch ci-dessous :

```python
from hub import Dataset
import torch

mnist = Dataset("activeloop/mnist")
# conversion de MNIST au format PyTorch
mnist = mnist.to_pytorch(lambda x: (x["image"], x["label"]))

train_loader = torch.utils.data.DataLoader(mnist, batch_size=1, num_workers=0)

for image, label in train_loader:
    # Training loop here
```

### Créer un ensemble de données local
Si vous souhaitez travailler sur vos propres données au niveau local, vous pouvez commencer par créer un ensemble de données :
```python
from hub import Dataset, schema
import numpy as np

ds = Dataset(
    "./data/dataset_name",  # le chemin d'accès à l'ensemble de données
    shape = (4,),  # suivi de la convention du format numpy
    mode = "w+",  # mode de lecture et d'écriture
    schema = {  # blocs de données nommés qui peuvent spécifier différents types
    # Tensor est une structure générique qui peut contenir tout type de données
        "image": schema.Tensor((512, 512), dtype="float"),
        "label": schema.Tensor((512, 512), dtype="float"),
    }
)

# remplir les conteneurs de données avec des données (ici - des zéros pour l'inialisation)
ds["image"][:] = np.zeros((4, 512, 512))
ds["label"][:] = np.zeros((4, 512, 512))
ds.flush()  # exécution de la création de l'ensemble de données
```

Vous pouvez également préciser `s3://bucket/path`, `gcs://bucket/path` ou un lien azure. [Ici](https://docs.activeloop.ai/en/latest/simple.html#data-storage) vous pouvez trouver plus d'informations sur le stockage Cloud.
De plus, si vous avez besoin d'un ensemble de données accessible au public que vous ne trouvez pas dans le Hub, vous pouvez [déposer une demande](https://github.com/activeloopai/Hub/issues/new?assignees=&labels=i%3A+enhancement%2C+i%3A+needs+triage&template=feature_request.md&title=[FEATURE]+New+Dataset+Required%3A+%2Adataset_name%2A). Nous l'activerons pour tout le monde dès que nous le pourrons !

###  Téléchargez votre ensemble de données et accédez-y à partir de <ins>n'importe où</ins> en 3 étapes simples

1. Enregistrez un compte gratuit sur [Activeloop](https://app.activeloop.ai/register/?utm_source=github&utm_medium=repo&utm_campaign=readme) et s'authentifier localement:
```sh
activeloop register
activeloop login

# alternativement, ajouter le nom d'utilisateur et le mot de passe comme arguments (utilisation sur des plateformes comme Kaggle)
activeloop login -u username -p password
```

2. Créez ensuite un ensemble de données, en précisant son nom, et téléchargez-le sur votre compte. Par exemple :
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

3. Vous pouvez y accéder de n'importe où dans le monde, sur n'importe quel appareil disposant d'une ligne de commande :
```python
from hub import Dataset

ds = Dataset("username/dataset_name")
```


## Documentation

Pour des pipelines de données plus avancés, comme le téléchargement de grands ensembles de données ou l'application de nombreuses transformations, veuillez consulter notre [documentation](http://docs.activeloop.ai/?utm_source=github&utm_medium=repo&utm_campaign=readme).

## Tutoriels Notebooks
Le dossier [exemples](https://github.com/activeloopai/Hub/tree/master/examples) contient une série d'exemples et le dossier des [notebooks](https://github.com/activeloopai/Hub/tree/master/examples/notebooks) dispose de quelques Notebooks avec des exemples d'utilisation. Certains de ces Notebooks sont énumérés ci-dessous.

| Notebook  	|   Description	|   	|
|:---	|:---	|---:	|
| [Chargement d'images](https://github.com/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%201a%20-%20Uploading%20Images.ipynb) | Vue d'ensemble sur la manière de télécharger et de stocker des images sur le Hub |  [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%201a%20-%20Uploading%20Images.ipynb) |
| [Chargement de Dataframes](https://github.com/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%201b%20-%20Uploading%20Dataframes.ipynb)  	| Vue d'ensemble sur la façon de télécharger des Dataframes sur le Hub | [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%201b%20-%20Uploading%20Dataframes.ipynb)  	|
| [Chargement de ficher Audio](https://github.com/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%201c%20-%20Uploading%20Audio.ipynb) | Explique comment traiter les données audio dans le Hub |[![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%201c%20-%20Uploading%20Audio.ipynb) |
| [Récupération de données à distance](https://github.com/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%202%20-%20Retrieving%20Remote%20Data.ipynb) | Explique comment récupérer les données | [![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/activeloopai/Hub/blob/master/examples/tutorial/tutorial/Tutorial%202%20-%20Retrieving%20Remote%20Data.ipynb) |
| [Transformer les données](https://github.com/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%203%20-%20Transforming%20Data.ipynb) | Résumés sur la transformation des données avec Hub |[![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%203%20-%20Transforming%20Data.ipynb) |
| [Tensors Dynamiques](https://github.com/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%204%20-%20What%20are%20Dynamic%20Tensors.ipynb) | Manipuler des données de forme et de taille différentes |[![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%204%20-%20What%20are%20Dynamic%20Tensors.ipynb) |
| [NLP avec Hub](https://github.com/activeloopai/Hub/blob/master/examples/notebooks/nlp_using_hub.ipynb) | Fine Tuning Bert for CoLA|[![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/activeloopai/Hub/blob/master/examples/notebooks/nlp_using_hub.ipynb) |


## Exemples d'utilisation
* **Imagerie satellite et drone**: [Une agriculture plus intelligente grâce à des conduites aériennes modulables](https://activeloop.ai/usecase/intelinair?utm_source=github&utm_medium=repo&utm_campaign=readme), [Cartographie du bien-être économique en Inde](https://towardsdatascience.com/faster-machine-learning-using-hub-by-activeloop-4ffb3420c005), [Combattre le criquet pèlerin au Kenya avec la Croix-Rouge](https://omdena.com/projects/ai-desert-locust/)
* **Images médicales**: Images volumétriques telles des IRM ou Xray
* **Voitures autonomes**: [Radar, LIDAR 3D, nuage de points, segmentation sémantique, objets vidéo](https://medium.com/snarkhub/extending-snark-hub-capabilities-to-handle-waymo-open-dataset-4dc7b7d8ab35)
* **Vente au détail**: Ensembles de données de caisses en libre-service
* **Media**: Stockage d'images, de vidéos, et fichiers audio 

## Pourquoi Hub en particulier ?

Il existe un certain nombre de bibliothèques de gestion d'ensembles de données qui offrent des fonctionnalités qui peuvent sembler similaires à celles de Hub. En fait, un certain nombre d'utilisateurs migrent des données de PyTorch ou de Tensorflow Datasets vers Hub. Voici quelques surprenantes différences que vous rencontrerez après avoir basculé vers Hub :
* les données sont fournies par morceaux, que vous pouvez diffuser en continu à distance, au lieu de les télécharger toutes en une fois
* comme seule la partie nécessaire de l'ensemble de données est évaluée, vous pouvez travailler sur les données immédiatement
* vous êtes en mesure de stocker les données qui n'entreraient pas dans votre mémoire dans son intégralité
* vous pouvez contrôler les versions et collaborer avec plusieurs utilisateurs sur vos ensembles de données sur différentes machines
* vous êtes équipé d'outils qui vous permettent de mieux comprendre les données en quelques secondes, comme notre outil de visualisation
* vous pouvez facilement préparer vos données pour plusieurs bibliothèques de ML dans une seule (par exemple, vous pouvez utiliser le même ensemble de données pour le training avec PyTorch et Tensorflow)

## communauté

Rejoignez notre [**Slack communauté**](https://join.slack.com/t/hubdb/shared_invite/zt-ivhsj8sz-GWv9c5FLBDVw8vn~sxRKqQ) pour obtenir l'aide de l'équipe Activeloop et d'autres utilisateurs, ainsi que pour se tenir au courant des meilleures pratiques en matière de gestion et de prétraitement des ensembles de données.

<img alt="tweet" src="https://img.shields.io/twitter/follow/activeloopai?label=stay%20in%20the%20Loop&style=social"> sur Twitter.

Comme toujours, merci à nos formidables contributeurs !    

<a href="https://github.com/activeloopai/hub/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=activeloopai/hub" />
</a>

Réalisé avec [contributors-img](https://contrib.rocks).

Veuillez lire [CONTRIBUTING.md](CONTRIBUTING.md) pour savoir comment commencer à apporter des contributions à Hub.
## Examples
Le format Hub d'Activeloop vous permet d'obtenir une inférence plus rapide à un coût moindre. Nous avons déjà plus de 30 ensembles de données populaires sur notre plateforme. Parmi ceux-ci, on peut citer
- COCO
- CIFAR-10
- PASCAL VOC
- Cars196
- KITTI
- EuroSAT 
- Caltech-UCSD Birds 200
- Food101

Consultez ces ensembles de données et bien d'autres encore sur notre [application web de visualisation](https://app.activeloop.ai/datasets/popular/?utm_source=github&utm_medium=repo&utm_campaign=readme) et chargez-les directement pour le training des modèles !

## Badge README 

Vous utilisez Hub ? Ajoutez un badge README pour que tout le monde soit au courant : 

[![hub](https://img.shields.io/badge/powered%20by-hub%20-ff5a1f.svg)](https://github.com/activeloopai/Hub)

```
[![hub](https://img.shields.io/badge/powered%20by-hub%20-ff5a1f.svg)](https://github.com/activeloopai/Hub)
```

## Avertissements

Comme d'autres progiciels de gestion d'ensembles de données, "Hub" est une bibliothèque utilitaire qui télécharge et prépare des ensembles de données publiques. Nous n'hébergeons ni ne distribuons ces ensembles de données, nous ne garantissons ni leur qualité ni leur équité, et nous ne prétendons pas que vous avez une licence d'utilisation de l'ensemble de données. Il est de votre responsabilité de déterminer si vous avez l'autorisation d'utiliser l'ensemble de données dans le cadre de la licence d'utilisation de l'ensemble de données.

Si vous êtes propriétaire d'un ensemble de données et que vous souhaitez en mettre à jour une partie (description, citation, etc.), ou si vous ne souhaitez pas que votre ensemble de données soit inclus dans cette bibliothèque, veuillez nous contacter par le biais d'un [ticket GitHub](https://github.com/activeloopai/Hub/issues/new). Merci pour votre contribution à la communauté ML !


## Acknowledgement
Cette technologie a été inspirée par notre expérience à l'université de Princeton et nous tenons à remercier William Silversmith @SeungLab pour son formidable outil [cloud-volume](https://github.com/seung-lab/cloud-volume). Nous sommes de grands utilisateurs de [Zarr](https://zarr.readthedocs.io/en/stable/) et nous voudrions remercier leur communauté pour avoir construit un bloc fondamental aussi important. 

