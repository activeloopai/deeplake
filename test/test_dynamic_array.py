import hub

conn = hub.s3(
            bucket='waymo-dataset-upload', 
            aws_creds_filepath='.creds/aws.json'
            ).connect()

arr = conn.array_create('test/dynamic_array_1', shape=(10000, 11, 400, 7), chunk=(100, 11, 400, 7), dtype='uint8', dsplit=2)
print(arr.darray.shape)
print(arr[6, 2].shape)
arr.darray[0:100, 0:5] = (200, 6)
arr.darray[0:100, 5:11] = (300, 4)

print(arr[5, 3, :, :].shape)
print(arr[5, 6, :, :].shape)
print(arr[5, 4:6, :, :].shape)
