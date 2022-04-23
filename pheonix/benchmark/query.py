import hub
import time

if __name__ == "__main__":
    # copy the dataset to local machine
    hub_path = "hub://activeloop/imagenet-train"
    #hub_path = "./data/cifar100"
    #hub_path = "hub://activeloop/cifar100-train"
    if not hub.exists("./data/cifar100"):
        hub.copy("hub://activeloop/cifar100-train", "./data/cifar100")
    
    t1 = time.time()
    ds = hub.load(hub_path)
    t2 = time.time()
    ds.filter("labels==0 and labels==1", progressbar=True)
    t3 = time.time()
    print("Loading dataset: ", t2-t1, "querying: ", t3-t2)