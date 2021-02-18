# Hub's Development Roadmap
## Overview
We are introducing this public development roadmap to help Hub users and contributors understand the project's direction. For a longer discussion, please refer to this [document](https://www.notion.so/Hub-Roadmap-24854a6648bb488e9063681890f8c890). It is meant for anyone interested in Hub, including developers and users.

## Organizational Structure
Features fall into one of two categories: "Data In" and "Data Out". Ideally, "Data In" features should improve usability and accessibility while "Data Out" features should improve benchmark performance.

### Data In
"Data In" features help users push data to Hub. New features should improve usability.

Currently, the process consists of:
* Fetching an existing dataset from a remote url, verifying dataset integrity (sha-256), unzipping dataset
* Identifying organizational structure of dataset (this is not easily automated since there isn't a standard way to organize large datasets)
* Defining Hub schema for dataset (which can be challenging if the abstract data structure does not exist already, eg DataFrame)
* Pushing dataset to a Hub repo (or another location, whether local or another remote store)

We would like to abstract away as many steps as possible. Examples include [schema generation](https://github.com/activeloopai/Hub/pull/344) and [higher level dataset objects](https://github.com/activeloopai/Hub/issues/505).

### Data Out
"Data Out" features help users stream data from Hub. New features should meaningfully improve performance (benchmarking scripts can be found [here](https://github.com/activeloopai/Hub/tree/master/benchmarks).

Currently, the process consists of:
* Locating the relevant dataset on Hub
* Applying transformations (perhaps PyTorch transforms)
* Fetching relevant slices from dataset (ie for some downstream task, such as model training)

Examples include [tokenization](https://github.com/activeloopai/Hub/issues/503) and [subsampling](https://github.com/activeloopai/Hub/issues/513).


## Contributing to the roadmap
### Add a new project for consideration
Anyone from the community can add a new proposed project to the roadmap:
> 1. By adding a [new issue](https://github.com/activeloopai/Hub/issues) in the repo
> 1. Add the issue to the [project board](https://github.com/activeloopai/Hub/projects/10)
 
### Prioritization
When a new project is created, it will be placed in the **Discussion** column, where it receives feedback from the community and core maintainers (@AbhinavTuli, @davidbuniat, @edogrigqv2, @kristinagrig06, @haiyangdeperci, @mynameisvinn, @mikayelh).

For a project to happen (in other words, get "prioritized" in the roadmap), three things need to be in place:

> 1. The scope is clear enough to understand the functional benefits and user impact on a high level
> 1. A community member (ideally the proposer) is willing to dedicate the effort and resources to make the project happen
> 1. A core maintainer is willing to sponsor the project - that is, define scope, write corresponding unit tests, and review PR

When these conditions are met, the card is moved to the **Committed** column. The subsequent **In Development** and **Done** columns indicate the development status.