import hub.auto.computer_vision.classification as auto

dss = auto.multiple_image_parse(
    "hub/auto/computer_vision/data/dataset", scheduler="single", workers=2
)

print(dss)

for i in dss:
    i.upload(i, "darkdebo/testing")
