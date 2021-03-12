Install this branch of Hub locally using pip3 install -e . instead of installing hub from pypi as this branch has a few changes not in master currently

ensure that the AWS creds are present in the enviroment

explore.ipynb should give you an idea of how Hub is being used for training and should also let you explore the dataset

dataset_generation.py contains the transform code that we adapted to hub format from the tfds code provided

model_training folder consists largely of the code from the repo https://github.com/brucechou1983/CheXNet-Keras adapted to use Hub. Only train.py and confi.ini should be relevant. 