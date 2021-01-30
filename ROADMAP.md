# Hub's Development Roadmap
## Overview
We are introducing this public development roadmap to help our users and contributors understand the project's direction. For a longer discussion, please refer to this [document](https://www.notion.so/Hub-Roadmap-24854a6648bb488e9063681890f8c890).

### Target audience
Anyone interested in Hub. This includes developers and end users

## Organizational Structure
Features for the roadmap fall into one of two categories.
### Data In
Features in this category help users - especially beginners -  push data to Hub. Examples include automatic schema generation, new dataset schemas, download scripts.

Currently, the process consists of:
* downloading dataset from remote url
* verifying dataset integrity (sha-256)
* unzipping dataset
* identifying organizational structure of dataset (there is no consistent way to organize large datasets)
* defining Hub schema for dataset (which can be challenging if the abstract data structure does not exist already, eg DataFrame)
* pushing dataset to a Hub location
* verifying success
We would like to abstract away as many steps from users as possible.

### Data Out
Features in this category help users pull data out of Hub for downstream tasks (eg model training). Examples include tokenization and sub-sampling. 

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