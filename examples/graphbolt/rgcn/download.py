import argparse, tarfile, hashlib, os, shutil
from tqdm import tqdm
import urllib.request as ur
import numpy as np

GBFACTOR = float(1 << 30)

def decide_download(url):
    d = ur.urlopen(url)
    size = int(d.info()["Content-Length"])/GBFACTOR
    ### confirm if larger than 1GB
    if size > 1:
        return input("This will download %.2fGB. Will you proceed? (y/N) " % (size)).lower() == "y"
    else:
        return True
    

dataset_urls = {
    'homogeneous' : {
        'tiny' : 'https://igb-public-awsopen.s3.amazonaws.com/igb-homogeneous/igb_homogeneous_tiny.tar.gz',
        'small' : 'https://igb-public-awsopen.s3.amazonaws.com/igb-homogeneous/igb_homogeneous_small.tar.gz',
        'medium' : 'https://igb-public-awsopen.s3.amazonaws.com/igb-homogeneous/igb_homogeneous_medium.tar.gz'
    },
    'heterogeneous' : {
        'tiny' : 'https://igb-public-awsopen.s3.amazonaws.com/igb-heterogeneous/igb_heterogeneous_tiny.tar.gz',
        'small' : 'https://igb-public-awsopen.s3.amazonaws.com/igb-heterogeneous/igb_heterogeneous_small.tar.gz',
        'medium' : 'https://igb-public-awsopen.s3.amazonaws.com/igb-heterogeneous/igb_heterogeneous_medium.tar.gz'
    }  
}


md5checksums = {
    'homogeneous' : {
        'tiny' : '34856534da55419b316d620e2d5b21be',
        'small' : '6781c699723529902ace0a95cafe6fe4',
        'medium' : '4640df4ceee46851fd18c0a44ddcc622'
    },
    'heterogeneous' : {
        'tiny' : '83fbc1091497ff92cf20afe82fae0ade',
        'small' : '2f42077be60a074aec24f7c60089e1bd',
        'medium' : '7f0df4296eca36553ff3a6a63abbd347'
    }  
}


def check_md5sum(dataset_type, dataset_size, filename):
    original_md5 = md5checksums[dataset_type][dataset_size]

    with open(filename, 'rb') as file_to_check:
        data = file_to_check.read()    
        md5_returned = hashlib.md5(data).hexdigest()

    if original_md5 == md5_returned:
        print(" md5sum verified.")
        return
    else:
        os.remove(filename)
        raise Exception(" md5sum verification failed!.")
        

def download_dataset(path, dataset_type, dataset_size):
    output_directory = path
    url = dataset_urls[dataset_type][dataset_size]
    if decide_download(url):
        data = ur.urlopen(url)
        size = int(data.info()["Content-Length"])
        chunk_size = 1024*1024
        num_iter = int(size/chunk_size) + 2
        downloaded_size = 0
        filename = path + "/igb_" + dataset_type + "_" + dataset_size + ".tar.gz"
        with open(filename, 'wb') as f:
            pbar = tqdm(range(num_iter))
            for i in pbar:
                chunk = data.read(chunk_size)
                downloaded_size += len(chunk)
                pbar.set_description("Downloaded {:.2f} GB".format(float(downloaded_size)/GBFACTOR))
                f.write(chunk)
    print("Downloaded" + " igb_" + dataset_type + "_" + dataset_size, end=" ->")
    check_md5sum(dataset_type, dataset_size, filename)
    file = tarfile.open(filename)
    file.extractall(output_directory)
    file.close()
    size = 0
    for path, dirs, files in os.walk(output_directory+ "/" + dataset_size):
        for f in files:
            fp = os.path.join(path, f)
            size += os.path.getsize(fp)
    print("Final dataset size {:.2f} GB.".format(size/GBFACTOR))
    os.remove(filename)
    os.rename(output_directory+ "/" + dataset_size, output_directory+ "/" + "igb-" + dataset_type + "-" + dataset_size)
    return output_directory+ "/" + "igb-" + dataset_type + "-" + dataset_size


def split_data(label_path, set_dir):
    labels = np.load(label_path)

    total_samples = len(labels)
    print(total_samples)
    train_end = int(0.8 * total_samples)
    print(train_end)
    validation_end = int(0.9 * total_samples)
    print(validation_end)

    indices = np.arange(total_samples)
    train_indices = indices[:train_end]
    validation_indices = indices[train_end:validation_end]
    test_indices = indices[validation_end:]
    print(indices)
    print(train_indices)
    print(validation_indices)
    print(test_indices)

    train_labels = labels[:train_end]
    validation_labels = labels[train_end:validation_end]
    test_labels = labels[validation_end:]
    print(train_labels, len(train_labels))
    print(validation_labels,len(validation_labels))
    print(test_labels, len(test_labels))

    np.save(f"{set_dir}/train_indices.npy", train_indices)
    np.save(f"{set_dir}/validation_indices.npy", validation_indices)
    np.save(f"{set_dir}/test_indices.npy", test_indices)
    np.save(f"{set_dir}/train_labels.npy", train_labels)
    np.save(f"{set_dir}/validation_labels.npy", validation_labels)
    np.save(f"{set_dir}/test_labels.npy", test_labels)


def convert_npy2csv(mapping, source, dest):
    for key, values in mapping.items():
        npy_path = source + "/" + key + "/" + "edge_index.npy"
        np_array = np.load(npy_path)
        print(np_array, len(np_array))
        
        csv_path = dest + "/" + values[0] + ".csv"
        np.savetxt(csv_path, np_array, delimiter=',', fmt='%d')

        reversed_np_arr = np_array[:, ::-1]
        print(reversed_np_arr, len(reversed_np_arr))
        rev_csv_path = dest + "/" + values[1] + ".csv"
        np.savetxt(rev_csv_path, reversed_np_arr, delimiter=',', fmt='%d')


def process_dataset(path):
    
    # Make the directory for processed dataset
    processed_dir = path + "-seeds"
    os.makedirs(processed_dir, exist_ok=True)

    original_path = path + "/" + "processed"

    # Step 1: Copy Node (Node feature and label)
    node_dir = processed_dir + "/" + "data"
    os.makedirs(node_dir, exist_ok=True)
    source_dir = original_path + "/" + "paper"
    feature_file = source_dir + "/" + "node_feat.npy"
    label_file = source_dir + "/" + "node_label_19.npy"
    shutil.copy(feature_file, node_dir)
    shutil.copy(label_file, node_dir)

    # Step 2: Create labels
    set_dir = processed_dir + "/" + "set"
    os.makedirs(set_dir, exist_ok=True)
    split_data(label_file, set_dir)

    # Step 3: Convert necessary npy files to csv files 
    edge_dir = processed_dir + "/" + "edges"
    os.makedirs(edge_dir, exist_ok=True)
    mapping = {
        "paper__cites__paper" : ["cites", "cited_by"],
        "paper__written_by__author" : ["written_by", "writes"],
        "paper__topic__fos" : ["has_topic", "rev_has_topic"],
        "author__affiliated_to__institute" : ["affiliated_with", "rev_affiliated_with"]
    }
    convert_npy2csv(mapping, original_path, edge_dir)
    shutil.rmtree(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='datasets/', 
        help='path containing the datasets')
    parser.add_argument('--dataset_type', type=str, default='heterogeneous',
        choices=['homogeneous', 'heterogeneous'], 
        help='dataset type')
    parser.add_argument('--dataset_size', type=str, default='tiny',
        choices=['tiny', 'small', 'medium'], 
        help='size of the datasets')
    args = parser.parse_args()    
    path = download_dataset(args.path, args.dataset_type, args.dataset_size)
    process_dataset(path)
