import hub

# Connect to Hub
waymo = hub.gs('waymo_open_dataset_snark').connect() 

## Open an array
camera = waymo.array_open('v1/validation/images')
print(camera.shape)

## Get the first camera image from the first camera and compute mean
print(camera[0,0].mean())


# Open the dataset
ds = waymo.dataset_open('v1/training')
ds_val = waymo.dataset_open('v1/training')

# Get all arrays from the dataset
images, lasers, camera_projected, labels = ds['images'], ds['lasers_range_image'], ds['lasers_camera_projection'], ds['labels']
print(images.shape, lasers.shape, camera_projected.shape, labels.shape)

