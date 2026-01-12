import deeplake
import os
import simplejson as json
import argparse
import concurrent.futures
import gc


datasets_home = "/home/ubuntu/datasets/"
dataset_name_prefix = "bluesky_"
dataset_1m_path = f"{datasets_home}{dataset_name_prefix}1m_copy"
dataset_10m_path = f"{datasets_home}{dataset_name_prefix}10m_copy"
dataset_100m_path = f"{datasets_home}{dataset_name_prefix}100m_copy"
dataset_1000m_path = f"{datasets_home}{dataset_name_prefix}1000m_copy"

datasets = {
    1: dataset_1m_path,
    2: dataset_10m_path,
    3: dataset_100m_path,
    4: dataset_1000m_path
}

data_home = "/home/ubuntu/data/bluesky/"

# generate filename from number, example 1 - file_0001.json.json
def generate_filename(number):
    return f"file_{number:04d}.json.json"

generate_filenames = lambda n: [generate_filename(i) for i in range(1, n + 1)]

# has two arguments: recreate - boolean, and a single unnamed number for the dataset
def parse_arguments():
    parser = argparse.ArgumentParser(description="Ingest Bluesky data into DeepLake")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate the dataset if it already exists",
    )
    parser.add_argument(
        "dataset",
        type=int,
        help="The dataset number to ingest",
    )
    args = parser.parse_args()
    return args

def should_recreate():
    args = parse_arguments()
    return args.recreate

def create_dataset(path):
    if should_recreate():
        if os.path.exists(path):
            ds = deeplake.delete(path)
    ds = deeplake.create(path)
    ds.add_column("json", deeplake.types.Dict())
    ds.commit()
    return ds

def read_file(filename):
    lines = []
    try:
        with open(os.path.join(data_home, filename), 'r', encoding='utf-8') as f:
            i = 0;
            buffer = ''
            for line in f:
                i += 1
                buffer += line.strip()
                try:
                    lines.append(json.loads(line.strip()))
                    buffer = ''
                except json.JSONDecodeError as e:
                    print("parse error: ", e)
                    buffer += '\n'
                #if i > 15: return lines
        return lines
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None

def ingest_single_file(ds, filename):
    print("reading file: ", filename)
    data = read_file(filename)
    if data:
        print("finished reading the file, appending...")
        ds.append({"json": data})
        data = None  # free memory
        print("finished appending, commiting...")
        ds.commit()
        print("finished commiting")
        ds.summary()

def ingest_files(ds, filenames):
    # Process files in batches of 4 to control memory usage
    batch_size = 15
    for i in range(0, len(filenames), batch_size):
        batch = filenames[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1} of {(len(filenames) + batch_size - 1)//batch_size}")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor:
            # Create futures list
            futures = []
            for filename in batch:
                future = executor.submit(ingest_single_file, ds, filename)
                futures.append(future)
                
            # Wait for all futures to complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing file in batch: {e}")
                    
        # Force garbage collection after each batch
        #gc.collect()
    
    return ds

def generate_dataset(number):
    if number not in datasets:
        print(f"Dataset {number} does not exist.")
        return
    files_count = {1: 1, 2: 10, 3: 100, 4: 1000}
    ds = create_dataset(datasets[number])
    ds = ingest_files(ds, generate_filenames(files_count[number]))
    ds.summary()

if __name__ == "__main__":
    args = parse_arguments()
    generate_dataset(args.dataset)
