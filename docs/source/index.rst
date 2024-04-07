.. deeplake documentation master file, created by
   sphinx-quickstart on Sun Jul 17 09:57:51 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/activeloopai/deeplake

Deep Lake API Reference
=======================

Deep Lake is an open-source database for AI. This page contains the Deep Lake API reference, mostly focusing on codebase documentation.
For tutorials, code examples, and high-level information, please check out the `Deep Lake Docs <https://docs.activeloop.ai/>`_.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   
   Installation

.. toctree::
   :maxdepth: 2
   :caption: Key Concepts

   Datasets
   Vector Store <Vector-Store>
   Tensors
   Htypes
   Compressions <Compressions>
   PyTorch and Tensorflow Support <Pytorch-and-Tensorflow-Support>
   Utility Functions <Utility-Functions>

.. toctree::
   :caption: Integrations

   Weights and Biases <Weights-and-Biases>

   MMDetection <MMDetection>

.. toctree::
   :maxdepth: 1
   :caption: High-Performance Features

   Dataloader <Dataloader>
   Sampler <Sampler>
   Tensor Query Language <Tensor-Query-Language>
   Random Split <Random-Split>
   Deep Memory <Deep-Memory>

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   deeplake
   deeplake.VectorStore
   deeplake.core
   deeplake.core.dataset
   deeplake.core.tensor
   deeplake.api
   deeplake.auto
   deeplake.util
   deeplake.client.log
   deeplake.core.transform
   deeplake.core.vectorstore.deep_memory
   deeplake.random


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
