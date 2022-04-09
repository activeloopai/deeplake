import hub

#TODO do wiki100, small-imagenet
#TODO add proper reporting/profiling

if __name__ == "__main__":
    
    # copy the dataset to local machine
    if not hub.exists("./data/cifar100"):
        hub.copy("hub://activeloop/cifar100-train", "./data/cifar100")
    
    ds = hub.load("./data/cifar100")
        
    for el in ds.pytorch(num_workers=0, use_progress_bar=True):
        pass 
