---
title: Hyperparameter Search
---
# Hyperparameter Search

It is possible to sample from discrete or continous range of parameters. You would need to provide the sampling method, number of samples, number of parallel workers and variables in the command. Params are templated in commands using double handlebars `{{}}`.

```yaml
version: 1
experiments:
  mnist_hyperparam_search:
    image: pytorch/pytorch:latest
    parameters:
      github: https://github.com/pytorch/examples
      batch_size: [32,64,128,256]
      lr:
        range: 0.1-0.3
        sampling: uniform
    hardware:
      gpu: k80
      gpu_count: 1
    samples: 4
    workers: 4
    command:
      - mkdir /demo && cd /demo
      - git clone {{github}} && cd examples/mnist
      - python main.py --batch-size {{batch_size}} --lr {{lr}}
```

The above script will start 4 instances and generate 4 tasks by randomly sampling from cartesian product of discrete lists and uniformly sampled continuous ranges. Each task will be executed on a machine. Each task is uniquely defined by the parameters.

For example one execution will run the following bash script on the instance.

```yaml
...
commands:
  - git clone {{github}} && cd examples/mnist
  - python main.py --batch-size {{batch_size}} --lr {{lr}}
```

into this one

```bash
- mkdir /demo && cd /demo
- git clone https://github.com/pytorch/examples && cd examples/mnist
- python main.py --batch-size 32 --lr 0.23
```
