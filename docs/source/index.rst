.. hub documentation master file, created by
   sphinx-quickstart on Wed Aug 21 21:40:09 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Hub's documentation!
===============================

Hub Array is a scalable numpy-like array stored on the cloud accessible over network as if they're local numpy arrays. 
You can create a Hub Array on your local environment as large as PetaBytes, use it as a local numpy array without worrying if the local disk/RAM will hold it or not. 
The Array is created on an object storage on the cloud and cached partially on your local environment. 
All read/write to the Array is automatically synchronized to the bucket.

Hub Array aims to provide a cloud-based data management solution for deep learning practitioners. 
Key features will include version control, feature store, data sharing and visualization for computer vision & NLP tasks, 

    **Hub Arrays**: scalable numpy-like arrays stored on the cloud accessible over internet 
    as if they're local numpy arrays.

Let's see how it works in action:

.. code-block:: bash 

   pip3 install hub

Create a large array remotely on cloud with some parts cached locally. 
You can read/write from anywhere as if it's a local array!

.. code-block:: python

    import hub
    bigarray = hub.array((10000000000, 512, 512, 3), 
                    name="test/bigarray:v0")

Documentation
-------------

**Getting Started**

* :doc:`getting-started/why-hub`
* :doc:`getting-started/quick-overview`
* :doc:`getting-started/faq`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   getting-started/why-hub
   getting-started/quick-overview
   getting-started/faq

**User Guide**

* :doc:`user-guide/indexing`
* :doc:`user-guide/computation`
* :doc:`user-guide/reshaping`
* :doc:`user-guide/combining`
* :doc:`user-guide/io`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: User Guide

   user-guide/indexing
   user-guide/computation
   user-guide/reshaping
   user-guide/combining
   user-guide/io

**Help & reference**

* :doc:`references/api`
* :doc:`references/roadmap`
* :doc:`references/contributing`
* :doc:`references/related-projects`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Help & reference

   references/api
   references/roadmap
   references/contributing
   references/related-projects



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
