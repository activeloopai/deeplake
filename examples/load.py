import hub
from hub import Dataset

# ds = Dataset("s3://snark-hub-dev/public/davis/mnist-new")
path = "s3://snark-hub-dev/public/davis/mnist-new"

ds = hub.load(path)
