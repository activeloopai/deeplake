.. Hub Documentation documentation master file, created by
   sphinx-quickstart on Mon May 18 23:53:34 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

What is Hub?
==============
.. image:: https://raw.githubusercontent.com/snarkai/Hub/master/docs/logo/hub_logo.png


**The fastest way to access and manage datasets for PyTorch and TensorFlow**

Hub provides fast access to the state-of-the-art datasets for Deep Learning, enabling data scientists to manage them, build scalable data pipelines and connect to Pytorch and Tensorflow 

Problems with Current Workflows
---------------------------------

We realized that there are a few problems related with current workflow in deep learning data management through our experience of working with deep learning companies and researchers. Most of the time Data Scientists/ML researchers work on data management and preprocessing instead of doing modeling. Deep Learning often requires to work with large datasets. Those datasets can grow up to terabyte or even petabyte size. 

1. It is hard to manage data, version control and track. 

2. It is time-consuming to download the data and link with the training or inference code. 

3. There is no easy way to access a chunk of it and possibly visualize. 

Wouldn’t it be more convenient to have large datasets stored & version-controlled as single numpy-like array on the cloud and have access to it from any machine at scale?



.. toctree::
   :maxdepth: 3
   :caption: Overview
   
   installing.md
   developing.md
   why.md
   benchmarks.md
   api

.. toctree::
   :maxdepth: 3
   :caption: Getting Started

   simple.md

.. toctree::
   :maxdepth: 3
   :caption: Concepts

   concepts/features.md
   concepts/dataset.md
   concepts/transform.md
   concepts/versioning.md
   concepts/filtering.md

.. toctree::
   :maxdepth: 3
   :caption: Integrations

   integrations/pytorch.md
   integrations/tensorflow.md

.. toctree::
   :maxdepth: 3
   :caption: Community

   community.md
   
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

