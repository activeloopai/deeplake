from nuscenes.nuscenes import NuScenes
from pathos.multiprocessing  import ProcessPool
from PIL import Image
import numpy as np
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
import hub
import time

nusc = NuScenes(version='v1.0-trainval', dataroot='/home/davit/nutonomy', verbose=True)

pool = ProcessPool(nodes=16)

length = 40000

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def create_datasets():
    hubs = {}
    for name in ['RADAR_FRONT', 'RADAR_BACK_RIGHT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT']:
        hubs[name] = hub.array(
            (length, 18, 200), 
            dtype = np.float32,
            name = 'aptiv/nutonomy3:{}'.format(name),
            chunk_size = (1000, 18, 200)
        )
    for name in ['LIDAR_TOP']:
        hubs[name] = hub.array(
            (length, 4, 30000), 
            dtype = np.float32,
            name = 'aptiv/nutonomy3:{}'.format(name),
            chunk_size = (100, 4, 30000)
        )
    for name in ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']:
        hubs[name] = hub.array(
            (length, 900, 1600, 3), 
            dtype = np.uint8,
            name = 'aptiv/nutonomy3:{}'.format(name),
            chunk_size = (5, 900, 1600, 3)
        )
    dataset = hub.dataset(hubs, name='aptiv/nutonomy3:v1.0-trainval')
    return dataset

def upload_scenes():
    index = 0
    data = {}
    for scene in nusc.scene:
        frame = nusc.get('sample', scene['first_sample_token'])
        # For each frame
        while not frame['next'] == "":
            for sensor in frame['data']:
                if sensor not in data:
                    data[sensor] = []
                sensor_data = frame['data'][sensor]
                data[sensor].append((index, sensor, sensor_data))
            index += 1
            frame = nusc.get('sample', frame['next'])

    def get_data(sensor_data):
        sd_record = nusc.get('sample_data', sensor_data)
        sensor_modality = sd_record['sensor_modality']
        nsweeps = 1
        sample_rec = nusc.get('sample', sd_record['sample_token'])
        ref_chan = 'LIDAR_TOP'
        chan = sd_record['channel']
        if sensor_modality == 'lidar':    
            pc, times = LidarPointCloud.from_file_multisweep(nusc, sample_rec, chan, ref_chan, nsweeps=nsweeps)
            points = pc.points
            return points # (4, 24962) max 30000
        elif sensor_modality == 'radar':
            pc, times = RadarPointCloud.from_file_multisweep(nusc, sample_rec, chan, ref_chan, nsweeps=nsweeps)
            points = pc.points
            return points # (18, -1) # max 100
        elif sensor_modality == 'camera':
            data_path, boxes, camera_intrinsic = nusc.get_sample_data(sensor_data) #, box_vis_level=box_vis_level)
            data = Image.open(data_path)
            img = np.array(data)
            return img # (900, 1600, 3)

    def upload_data(tasks):
        def batch(iterable, n=1):
            l = len(iterable)
            for ndx in range(0, l, n):
                yield iterable[ndx:min(ndx + n, l)]

        name = tasks[0][1]
        x = hub.load('aptiv/nutonomy:{}'.format(name))
        for b in batch(tasks, n=x.chunk_shape[0]):
            t1 = time.time()
            y = np.zeros(x.chunk_shape, dtype=x.dtype)
            i = 0 
            for el in b:
                data = np.array(get_data(el[-1]))
                y[i, ..., :data.shape[-1]] = data
                i += 1
            x[b[0][0]:b[-1][0]+1] = y[:b[-1][0]-b[0][0]+1]
            t2 = time.time()
            print('{}: uploaded chunk {}-{} in {}s'.format(name, b[0][0], b[-1][0], t2-t1))
            
            
    hubs = create_datasets()  
    for sensor in data:
        print('uploading sensor {}'.format(sensor))
        pool.map(upload_data, list(batch(data[sensor], n=hubs[sensor].chunk_shape[0])))

    return


upload_scenes()