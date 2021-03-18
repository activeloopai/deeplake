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

<h3 align="center"> Introducing Data 2.0, powered by Hub. </br>The fastest way to store, access & manage datasets with version-control for PyTorch/TensorFlow.  Works locally or on any cloud. Scalable data pipelines.</h3>

---

[ English | [Français](./README_translations/README_FR.md) | [简体中文](./README_translations/README_CN.md) | [Türkçe](./README_translations/README_TR.md) | [한글](./README_translations/README_KR.md) | [Bahasa Indonesia](./README_translations/README_ID.md)]

<i> Note: the translations of this document may not be up-to-date. For the latest version, please check the README in English. </i>

### What is Hub for?

Software 2.0 needs Data 2.0, and Hub delivers it. Most of the time Data Scientists/ML researchers work on data management and preprocessing instead of training models. With Hub, we are fixing this. We store your (even petabyte-scale) datasets as single numpy-like array on the cloud, so you can seamlessly access and work with it from any machine. Hub makes any data type (images, text files, audio, or video) stored in cloud usable as fast as if it were stored on premise. With same dataset view, your team can always be in sync. 

Hub is being used by Waymo, Red Cross, World Resources Institute, Omdena, and others.

### Features 

* Store and retrieve large datasets with version-control
* Collaborate as in Google Docs: Multiple data scientists working on the same data in sync with no interruptions
* Access from multiple machines simultaneously
* Deploy anywhere - locally, on Google Cloud, S3, Azure as well as Activeloop (by default - and for free!) 
* Integrate with your ML tools like Numpy, Dask, Ray, [PyTorch](https://docs.activeloop.ai/en/latest/integrations/pytorch.html), or [TensorFlow](https://docs.activeloop.ai/en/latest/integrations/tensorflow.html)
* Create arrays as big as you want. You can store images as big as 100k by 100k!
* Keep shape of each sample dynamic. This way you can store small and big arrays as 1 array. 
* [Visualize](http://app.activeloop.ai/?utm_source=github&utm_medium=repo&utm_campaign=readme) any slice of the data in a matter of seconds without redundant manipulations

 <p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/activeloopai/Hub/master/docs/visualizer%20gif.gif" width="75%"/>
    </br>
Visualization of a dataset uploaded to Hub via app.activeloop.ai (free tool).

</p>


## Documentation

Please refer to our [wiki](https://github.com/activeloopai/Hub/wiki) to get started with hub and  for more advanced data pipelines like uploading large datasets or applying many transformations, refer to our [documentation](http://docs.activeloop.ai/?utm_source=github&utm_medium=repo&utm_campaign=readme).


## Community

Join our [**Slack community**](https://join.slack.com/t/hubdb/shared_invite/zt-ivhsj8sz-GWv9c5FLBDVw8vn~sxRKqQ) to get help from Activeloop team and other users, as well as stay up-to-date on dataset management/preprocessing best practices.

<img alt="tweet" src="https://img.shields.io/twitter/follow/activeloopai?label=stay%20in%20the%20Loop&style=social"> on Twitter.

As always, thanks to our amazing contributors!    

<a href="https://github.com/activeloopai/hub/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=activeloopai/hub" />
</a>

Made with [contributors-img](https://contrib.rocks).

Please read [CONTRIBUTING.md](CONTRIBUTING.md) to know how to get started with making contributions to Hub.

## Examples
Activeloop's Hub format lets you achieve faster inference at a lower cost. We have 30+ popular datasets already on our platform. These include:
- COCO
- CIFAR-10
- PASCAL VOC
- Cars196
- KITTI
- EuroSAT 
- Caltech-UCSD Birds 200
- Food101

Check these and many more popular datasets on our [visualizer web app](https://app.activeloop.ai/datasets/popular/?utm_source=github&utm_medium=repo&utm_campaign=readme) and load them directly for model training!

## README Badge

Using Hub? Add a README badge to let everyone know: 


[![hub](https://img.shields.io/badge/powered%20by-hub%20-ff5a1f.svg)](https://github.com/activeloopai/Hub)

```
[![hub](https://img.shields.io/badge/powered%20by-hub%20-ff5a1f.svg)](https://github.com/activeloopai/Hub)
```
## Usage Tracking
By default, we collect anonymous usage data using Bugout (here's the [code](https://github.com/activeloopai/Hub/blob/853456a314b4fb5623c936c825601097b0685119/hub/__init__.py#L24) that does it). It only logs Hub library's own actions and parameters, and no user/ model data is collected.

This helps the Activeloop team to understand how the tool is used and how to deliver maximum value to the community by building features that matter to you. You can easily opt-out of usage tracking during login.


## Disclaimers

Similarly to other dataset management packages, `Hub` is a utility library that downloads and prepares public datasets. We do not host or distribute these datasets, vouch for their quality or fairness, or claim that you have license to use the dataset. It is your responsibility to determine whether you have permission to use the dataset under the dataset's license.

If you're a dataset owner and wish to update any part of it (description, citation, etc.), or do not want your dataset to be included in this library, please get in touch through a [GitHub issue](https://github.com/activeloopai/Hub/issues/new). Thanks for your contribution to the ML community!


## Acknowledgement
 This technology was inspired from our experience at Princeton University and would like to thank William Silversmith @SeungLab with his awesome [cloud-volume](https://github.com/seung-lab/cloud-volume) tool. We are heavy users of [Zarr](https://zarr.readthedocs.io/en/stable/) and would like to thank their community for building such a great fundamental block. 
