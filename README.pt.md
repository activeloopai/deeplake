<img src="https://static.scarf.sh/a.png?x-pxid=bc3c57b0-9a65-49fe-b8ea-f711c4d35b82" />

<p align="center">
  <img src="https://user-images.githubusercontent.com/83741606/156426873-c0a77da0-9e0f-41a0-a4fb-cf77eb2fe35e.png" width="300"/>
  <h1 align="center">Formatação de Dados para IA</h1>
</p>

<p align="center">
    <a href="https://github.com/activeloopai/Hub/actions/workflows/test-pr-on-label.yml"><img src="https://github.com/activeloopai/Hub/actions/workflows/test-push.yml/badge.svg" alt="PyPI version" height="18"></a>
    <a href="https://pypi.org/project/hub/"><img src="https://badge.fury.io/py/hub.svg" alt="PyPI version" height="18"></a>
    <a href="https://pepy.tech/project/hub"><img src="https://static.pepy.tech/personalized-badge/hub?period=month&units=international_system&left_color=grey&right_color=orange&left_text=Downloads" alt="PyPI version" height="18"></a>
     <a href="https://github.com/activeloopai/Hub/issues">
    <img alt="GitHub issues" src="https://img.shields.io/github/issues/activeloopai/Hub"> </a>
    <a href="https://codecov.io/gh/activeloopai/Hub/branch/main"><img src="https://codecov.io/gh/activeloopai/Hub/branch/main/graph/badge.svg" alt="codecov" height="18"></a>
</p>

<h3 align="center">
   <a href="https://docs.activeloop.ai/?utm_source=github&utm_medium=github&utm_campaign=github_readme&utm_id=readme"><b>Documentação</b></a> &bull;
   <a href="https://docs.activeloop.ai/getting-started/?utm_source=github&utm_medium=github&utm_campaign=github_readme&utm_id=readme"><b>Primeiros Passos</b></a> &bull;
   <a href="https://api-docs.activeloop.ai/"><b>Referências da API</b></a> &bull;  
   <a href="https://github.com/activeloopai/examples/"><b>Exemplos</b></a> &bull;
   <a href="https://www.activeloop.ai/resources/?utm_source=github&utm_medium=github&utm_campaign=github_readme&utm_id=readme"><b>Blog</b></a> &bull;  
  <a href="http://slack.activeloop.ai"><b>Comunidade no Slack</b></a> &bull;
  <a href="https://twitter.com/intent/tweet?text=The%20dataset%20format%20for%20AI.%20Stream%20data%20to%20PyTorch%20and%20Tensorflow%20datasets&url=https://activeloop.ai/&via=activeloopai&hashtags=opensource,pytorch,tensorflow,data,datascience,datapipelines,activeloop,databaseforAI"><b>Twitter</b></a>
 </h3>

*Leia isto em outros idiomas: [简体中文](README.zh-cn.md), [Inglês](README.pt.md)*

# Conteúdos

<!-- TOC -->

- [Conteúdos](#conteúdos)
  - [ℹ️ Sobre a Hub](#ℹ️-sobre-a-hub)
  - [🚀 Dando os Primeiros Passos com a Hub](#-dando-os-primeiros-passos-com-a-hub)
    - [💻 Como Instalar o Hub](#-como-instalar-o-hub)
    - [🧠 Treinando um modelo PyTorch em um conjunto de dados da Hub](#-treinando-um-modelo-pytorch-em-um-conjunto-de-dados-da-hub)
      - [Carregar Cifar 10, um dos conjuntos de dados prontamente disponíveis no Hub](#carregar-cifar-10-um-dos-conjuntos-de-dados-prontamente-disponíveis-no-hub)
      - [Inspecione os tensors no conjunto de dados](#inspecione-os-tensors-no-conjunto-de-dados)
      - [Treine um modelo PyTorch no conjunto de dados Cifar 10 sem a necessidade de baixá-lo](#treine-um-modelo-pytorch-no-conjunto-de-dados-cifar-10-sem-a-necessidade-de-baixá-lo)
    - [🏗️ Como criar os Dados na Hub](#️-como-criar-os-dados-na-hub)
    - [🔄 Como carregar os Dados da Hub](#-como-carregar-os-dados-da-hub)
  - [📚 Documentação](#-documentação)
  - [🎓 Para Estudantes e Educadores](#-para-estudantes-e-educadores)
  - [👩‍💻 Comparações entre Ferramentas Familiares](#-comparações-entre-ferramentas-familiares)
  - [👨‍👩‍👧‍👦 Comunidade](#-comunidade)
  - [🔖 Emblema para o README](#-emblema-para-o-readme)
  - [🏛️ Avisos Legais](#️-avisos-legais)
  - [💬 citações](#-citações)
  - [✒️ Reconhecimento](#️-reconhecimento)

<!-- /TOC -->

## ℹ️ Sobre a Hub

O Hub é um formato de conjunto de dados com uma API simples para criar, armazenar e colaborar nos conjuntos de dados de AI de qualquer tamanho.Ele permite armazenar todos os seus dados em um só lugar, variando de anotações simples a vídeos grandes, e desbloqueia um fluxo rápido de dados ao treinar modelos em escala.O Hub é usado pelo Google, Waymo, Cruz Vermelha, Universidade de Oxford e Omdena.Hub inclui os seguintes recursos:

<details>
  <summary><b>Armazenamento agnóstico API</b></summary>
Use a mesma API para fazer upload, baixar e transmitir conjuntos de dados de/para o AWS S3/S3 Compatível de armazenamento, GCP, ActiveLoop Cloud, armazenamento local e também em memória.
</details>

<details>
  <summary><b>Armazenamento comprimido</b></summary>
Armazene imagens, áudios e vídeos em sua compressão nativa, descomprimindo -os apenas quando necessário, por exemplo, ao treinar um modelo.
</details>

<details>
  <summary><b>Indexação preguiçosa do tipo Numpy</b></summary>
Trate seus conjuntos de dados S3 ou GCP como se fossem uma coleção de matrizes Numpy na memória do seu sistema.Corte -os, indexe -os ou itera através deles.Somente os bytes que você pedir serão baixados!
</details>

<details>
  <summary><b>Controle da versão do conjunto de dados</b></summary>
Compromissos, Ramificações, Checkout. Conceitos Com Os Quais Você Já Está Familiarizado Em Seus Repositórios De Código Agora Pode Ser Aplicado Aos Seus Conjuntos De Dados Também!
</details>

<details>
  <summary><b>Integrações com estruturas de aprendizado profundo</b></summary>
Hub Vem com integrações internas para Pytorch e Tensorflow. Treine seu modelo com algumas linhas de código - até cuidamos do conjunto de dados. :)
</details>

<details>
  <summary><b>Transformações distribuídas</b></summary>
Aplique rapidamente transformações em seus conjuntos de dados usando multi-threading, multiprocessamento ou nosso interno <a href="https://www.ray.io/">Ray</a> integração.</details>

<details>
  <summary><b>100+ conjuntos de dados de imagem, vídeo e áudio mais populares disponíveis em segundos</b></summary>
Hub Comunidade enviou <a href="https://docs.activeloop.ai/datasets/?utm_source=github&utm_medium=github&utm_campaign=github_readme&utm_id=readme">100+ conjuntos de dados de imagem, vídeo e áudio</a> como <a href="https://docs.activeloop.ai/datasets/mnist/?utm_source=github&utm_medium=github&utm_campaign=github_readme&utm_id=readme">MNIST</a>, <a href="https://docs.activeloop.ai/datasets/coco-dataset/?utm_source=github&utm_medium=github&utm_campaign=github_readme&utm_id=readme">COCO</a>,  <a href="https://docs.activeloop.ai/datasets/imagenet-dataset/?utm_source=github&utm_medium=github&utm_campaign=github_readme&utm_id=readme">ImageNet</a>,  <a href="https://docs.activeloop.ai/datasets/cifar-10-dataset/?utm_source=github&utm_medium=github&utm_campaign=github_readme&utm_id=readme">CIFAR</a>,  <a href="https://docs.activeloop.ai/datasets/gtzan-genre-dataset/?utm_source=github&utm_medium=github&utm_campaign=github_readme&utm_id=readme">GTZAN</a> e outros.
</details>

<details>
  <summary><b>Suporte de visualização instantâneo na <a href="https://app.activeloop.ai/?utm_source=github&utm_medium=github&utm_campaign=github_readme&utm_id=readme">Plataforma ActiveLoop</a></b></summary>
Hub Os conjuntos de dados são visualizados instantaneamente com caixas delimitadoras, máscaras, anotações, etc. em <a href="https://app.activeloop.ai/?utm_source=github&utm_medium=github&utm_campaign=github_readme&utm_id=readme">Plataforma ActiveLoop</a> (Veja abaixo).
</details>

<div align="center">
<a href="https://www.linkpicture.com/view.php?img=LPic61b13e5c1c539681810493"><img src="https://www.linkpicture.com/q/ReadMe.gif" type="image"></a>
</div>

## 🚀 Dando os Primeiros Passos com a Hub

### 💻 Como Instalar o Hub

Hub está escrito em 100% Python e pode ser instalado rapidamente usando o PIP.

```sh
pip3 install hub
```

**Por padrão, o Hub não instala dependências para suporte de áudio, vídeo e Google-Cloud (GCS).Eles podem ser instalados usando**:

```sh
pip3 install "hub[av]"          -> Suporte de áudio e vídeo via PyAV
pip3 install "hub[gcp]"         -> GCS suporte via google-* dependências
pip3 install "hub[all]"         -> Instala tudo - áudio vídeo e suporte GCS 
```

### 🧠 Treinando um modelo PyTorch em um conjunto de dados da Hub

#### Carregar Cifar 10, um dos conjuntos de dados prontamente disponíveis no Hub

```python
import hub
import torch
from torchvision import transforms, models

ds = hub.load('hub://activeloop/cifar10-train')
```

#### Inspecione os tensors no conjunto de dados

```python
ds.tensors.keys()    # dict_keys(['images', 'labels'])
ds.labels[0].numpy() # array([6], dtype=uint32)
```

#### Treine um modelo PyTorch no conjunto de dados Cifar 10 sem a necessidade de baixá-lo

Primeiro, defina uma transformação para as imagens e use o Pytorch de uma linha embutido do Hub para conectar os dados à computação:

```python
tform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

hub_loader = ds.pytorch(num_workers=0, batch_size=4, transform={'images': tform, 'labels': None}, shuffle=True)
```

Em seguida, defina o modelo, perda e otimizador:

```python
net = models.resnet18(pretrained=False)
net.fc = torch.nn.Linear(net.fc.in_features, len(ds.labels.info.class_names))
    
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```

Finalmente, o loop de treinamento para 2 épocas:

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

### 🏗️ Como criar os Dados na Hub

Um conjunto de dados de hub pode ser criado em vários locais (provedores de armazenamento).É assim que os caminhos para cada um deles seriam:

| Provedor de armazenamento | Exemplo de caminho          |
| ---------------------- | ------------------------------ |
| Activeloop cloud       | hub://user_name/dataset_name   |
| AWS S3 / S3 compatível | s3://bucket_name/dataset_name  |
| GCP                    | gcp://bucket_name/dataset_name |
| Google Drive           | gdrive://path_to_dataset       |
| Armazenamento local    | caminho para o diretório local |
| In-memory              | mem://dataset_name             |

Vamos criar um conjunto de dados na nuvem ActiveLoop.ActiveLoop Cloud fornece armazenamento gratuito de até 300 GB por usuário (mais informações [aqui](#-for-students-and-educators)). Crie uma nova conta com o Hub a partir do terminal usando o `ActiveLoop Register`, se você ainda não o fez.Você será solicitado um nome de usuário, ID de email e senha.O nome de usuário que você inserir aqui será usado no caminho do conjunto de dados.

```sh
$ activeloop register
Enter your details. Your password must be at least 6 characters long.
Username:
Email:
Password:
```

Inicialize um conjunto de dados vazio na nuvem ActiveLoop:

```python
import hub

ds = hub.empty("hub://<USERNAME>/test-dataset")
```

Em seguida, crie um tensor para manter imagens no conjunto de dados que acabamos de inicializar:

```python
images = ds.create_tensor("images", htype="image", sample_compression="jpg")
```

Supondo que você tenha uma lista de caminhos de arquivo de imagem, vamos carregá-los no conjunto de dados:

```python
image_paths = ...
with ds:
    for image_path in image_paths:
        image = hub.read(image_path)
        ds.images.append(image)
```

Como alternativa, você também pode fazer upload de matrizes numpy. Como o tensor `images` foi criado com `sample_compression = "jpg"`, as matrizes serão compactadas com compressão JPEG.

```python
import numpy as np

with ds:
    for _ in range(1000):  # 1000 random images
        random_image = np.random.randint(0, 256, (100, 100, 3))  # 100x100 image with 3 channels
        ds.images.append(random_image)
```

### 🔄 Como carregar os Dados da Hub

Você pode carregar o conjunto de dados que acabou de criar com uma única linha de código:

```python
import hub

ds = hub.load("hub://<USERNAME>/test-dataset")
```

Você também pode acessar um dos <a href="https://docs.activeloop.ai/datasets/?utm_source=github&utm_medium=github&utm_campaign=github_readme&utm_id=readme">100+ conjuntos de dados de imagem, vídeo e áudio no formato do hub </a>, não apenas os que você criou.Aqui está como você carregaria o [Conjunto de dados de bicicletas objectron](https://github.com/google-research-datasets/Objectron):

```python
import hub

ds = hub.load('hub://activeloop/objectron_bike_train')
```

Para obter a primeira imagem no conjunto de dados do Objectron Bikes em formato Numpy:

```python
image_arr = ds.image[0].numpy()
```

## 📚 Documentação

Iniciar guias, exemplos, tutoriais, referência da API e outras informações úteis podem ser encontradas em nossa [página de documentação](http://docs.activeloop.ai/?utm_source=github&utm_medium=repo&utm_campaign=readme).

## 🎓 Para Estudantes e Educadores

Os usuários do hub podem acessar e visualizar uma variedade de conjuntos de dados populares por meio de uma integração gratuita com a plataforma da ActiveLoop.Os usuários também podem criar e armazenar seus próprios conjuntos de dados e disponibilizá -los ao público.O armazenamento gratuito de até 300 GB está disponível para estudantes e educadores:

| <!-- -->                                          | <!-- -->   |
| ------------------------------------------------- | ---------- |
| Armazenamento para conjuntos de dados públicos hospedados pela ActiveLoop  | 200GB grátis |
| Armazenamento para conjuntos de dados privados hospedados pela ActiveLoop | 100GB grátis |

## 👩‍💻 Comparações entre Ferramentas Familiares

<details>
  <summary><b>Activeloop Hub vs DVC</b></summary>
  
O Hub e o DVC oferecem controle de versão do conjunto de dados semelhante ao Git para dados, mas seus métodos para armazenar dados diferem significativamente.O Hub converte e armazena dados como matrizes compactadas em chunk, que permitem streaming rápido para modelos ML, enquanto o DVC opera sobre os dados armazenados em estruturas de arquivos tradicionais menos eficientes.O formato do hub facilita significativamente a versão do conjunto de dados em comparação com as estruturas de arquivos tradicionais por DVC quando os conjuntos de dados são compostos de muitos arquivos (ou seja, muitas imagens).Uma distinção adicional é que o DVC usa principalmente uma interface de linha de comando, enquanto o Hub é um pacote Python.Por fim, o Hub oferece uma API para conectar facilmente os conjuntos de dados a estruturas ML e outras ferramentas comuns de ML e permite a visualização instantânea do conjunto de dados por meio [Ferramenta de visualização do ActiveLoop](http://app.activeloop.ai/?utm_source=github&utm_medium=repo&utm_campaign=readme).

</details>

<details>
  <summary><b>Activeloop Hub vs TensorFlow Datasets (TFDS)</b></summary>
  
O Hub e o TFDS conectam perfeitamente os conjuntos de dados populares às estruturas ML.Os conjuntos de dados de hub são compatíveis com Pytorch e Tensorflow, enquanto os TFDs são compatíveis apenas com o TensorFlow.Uma diferença importante entre o Hub e o TFDS é que os conjuntos de dados hub são projetados para streaming da nuvem, enquanto o TFDS deve ser baixado localmente antes do uso.Como resultado, com o hub, pode -se importar conjuntos de dados diretamente dos conjuntos de dados do TensorFlow e transmiti -los para Pytorch ou TensorFlow.Além de fornecer acesso a conjuntos de dados populares disponíveis ao público, o Hub também oferece ferramentas poderosas para criar conjuntos de dados personalizados, armazená -los em uma variedade de provedores de armazenamento em nuvem e colaborar com outras pessoas via API simples.O TFDS está focado principalmente em fornecer ao público fácil acesso a conjuntos de dados geralmente disponíveis, e o gerenciamento de conjuntos de dados personalizados não é o foco principal.Um artigo de comparação completo pode ser encontrado [aqui](https://www.activeloop.ai/resources/tensor-flow-tf-data-activeloop-hub-how-to-implement-your-tensor-flow-data-pipelines-with-hub/).

</details>

<details>
  <summary><b>Activeloop Hub vs HuggingFace</b></summary>
O Hub e o HuggingFace oferecem acesso a conjuntos de dados populares, mas o Hub se concentra principalmente na visão computacional, enquanto o Huggingface se concentra no processamento de linguagem natural.Transformagens de Huggingface e outras ferramentas computacionais para PNL não são análogas aos recursos oferecidos pelo Hub.

</details>

<details>
  <summary><b>Activeloop Hub vs WebDatasets</b></summary>
O Hub e o WebDatasets oferecem um fluxo rápido de dados entre as redes.Eles têm velocidades de vapor quase idênticas, porque as solicitações de rede subjacentes e as estruturas de dados são muito semelhantes.No entanto, o Hub oferece acesso aleatório e arrastamento superiores, sua API simples está no Python em vez de linha de comando, e o Hub permite a indexação e modificação simples do conjunto de dados sem ter que recriá-lo.

</details>

## 👨‍👩‍👧‍👦 Comunidade

Junte-se ao nosso [**Comunidade Slack**](https://join.slack.com/t/hubdb/shared_invite/zt-ivhsj8sz-GWv9c5FLBDVw8vn~sxRKqQ) Para saber mais sobre o gerenciamento de conjunto de dados não estruturado usando o Hub e obter ajuda da equipe ActiveLoop e de outros usuários.

Adoraríamos seu feedback completando nossos 3 minutos [**survey**](https://forms.gle/rLi4w33dow6CSMcm9).

Como sempre, graças aos nossos incríveis colaboradores!

<a href="https://github.com/activeloopai/hub/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=activeloopai/hub" />
</a>

Feito com [contributors-img](https://contrib.rocks).

Por favor leia [CONTRIBUTING.md](CONTRIBUTING.md) Para começar a fazer contribuições para o hub.

## 🔖 Emblema para o README

Usando o hub? Adicione um emblema no seu README para que todos saibam:

[![hub](https://img.shields.io/badge/powered%20by-hub%20-ff5a1f.svg)](https://github.com/activeloopai/Hub)

```md
[![hub](https://img.shields.io/badge/powered%20by-hub%20-ff5a1f.svg)](https://github.com/activeloopai/Hub)
```

## 🏛️ Avisos Legais

<details>
  <summary><b>Licença dos dados</b></summary>

Os usuários do hub podem ter acesso a uma variedade de conjuntos de dados disponíveis ao público. Não hospedamos ou distribuímos esses conjuntos de dados, atestamos sua qualidade ou justiça ou afirmamos que você tem uma licença para usar os conjuntos de dados. É sua responsabilidade determinar se você tem permissão para usar os conjuntos de dados em sua licença.

Se você é proprietário de um conjunto de dados e não deseja que seu conjunto de dados seja incluído nesta biblioteca, entre em contato através de um [GitHub issue](https://github.com/activeloopai/Hub/issues/new). Obrigado por sua contribuição para a comunidade ML!

</details>

<details>
  <summary><b>Rastreamento de uso</b></summary>

Por padrão, coletamos dados de uso usando Bugout (Aqui está o [codigo](https://github.com/activeloopai/Hub/blob/853456a314b4fb5623c936c825601097b0685119/hub/__init__.py#L24) que faz isso). Ele não coleta dados do usuário que não sejam dados de endereço IP anonimizado e apenas registra as próprias ações da biblioteca do hub.Isso ajuda nossa equipe a entender como a ferramenta é usada e como criar recursos que importam para você!Depois de se registrar no ActiveLoop, os dados não são mais anônimos.Você sempre pode optar por não participar de relatórios usando o comando da CLI abaixo:

```sh
activeloop reporting --off
```

</details>

## 💬 citações

Se você usar o hub em sua pesquisa, cite o Activeloop usando:

```txt
@article{
  2022ActiveloopHub,
  title = {
    Hub: A Dataset Format for AI. A simple API for creating, storing, collaborating on AI datasets of any size & streaming them to ML frameworks at scale.
  },
  author = {
    Activeloop Developer Team
  },
  journal = {
    GitHub.
    Note: https://github.com/activeloopai/Hub
  },
  year = {
    2022
  }
}
```

## ✒️ Reconhecimento

Essa tecnologia foi inspirada em nosso trabalho de pesquisa na Universidade de Princeton.Gostaríamos de agradecer William Silversmith @SeungLab pela sua incrível ferramenta [cloud-volume](https://github.com/seung-lab/cloud-volume).
