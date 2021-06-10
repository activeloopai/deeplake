import hub

kaggle_credentials = {"username":"dyllan","key":"9aa25fa1ea48d3ec521828b118d6e0e0"}
ds = hub.from_kaggle(tag="coloradokb/dandelionimages", source="./datasets/dandelionimages/unstructured", 
        destination="./datasets/dandelionimages/structured",  kaggle_credentials=kaggle_credentials)