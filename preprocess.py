import os
import csv
from datetime import datetime, timezone, timedelta
import math
import numpy as np
from operator import itemgetter
import random
import sys
import csv
import argparse
import shutil

from helper_function import \
    compute_start_end, \
    process_gt_file, \
    process_imu_file, \
    generate_gt, \
    output_check
    

threshold_ts = 100

def save_data_to_csv(path_data, csv_file_path):
    with open(csv_file_path, 'w', newline='') as csvfile:
        fieldnames = list(path_data.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(len(path_data[fieldnames[0]])):
            writer.writerow({
                f: path_data[f][i] for f in fieldnames
            })


def process_path(folder_path, transform):
    timestamp_ms_start, timestamp_ms_end = compute_start_end(folder_path)
    
    #Retrieve gt
    gt_path = os.path.join(folder_path, 'Gt', 'vis.txt')
    gt_data = process_gt_file(gt_path, timestamp_ms_start, timestamp_ms_end)
    
    #Retrieve mag
    mag_path = os.path.join(folder_path, 'Sensor', 'imu.txt')
    full_mag_data = process_imu_file(mag_path, transform= transform)
    
    
    matched_idxes, syn_gt_data, syn_mag_data = generate_gt(gt_data, full_mag_data)
    output_check(matched_idxes, full_mag_data, gt_data, syn_gt_data, syn_mag_data, threshold_ts)
    
    return syn_gt_data, syn_mag_data


def train_test_split(process_dir, test_ratio=0.1):
    folders = [f for f in os.listdir(process_dir) if os.path.isdir(os.path.join(process_dir, f))]
    random.shuffle(folders)
    split_index = int(len(folders) * (1 - test_ratio))
    
    train_folders = folders[:split_index]
    test_folders = folders[split_index:]
    
    return train_folders, test_folders

def main(args):    
    root_dir = args.root_dir
    path_num = args.path_num
    process_dir = os.path.join(root_dir, path_num)
    
    transform_yes = bool(args.transform)
    transform = 'transformed' if transform_yes else 'no_transform'

    # Perform train-test split
    train_folders, test_folders = train_test_split(process_dir, args.test_ratio)

    for split, folders in [('train', train_folders), ('test', test_folders)]:
        for folder_name in folders:
            folder_path = os.path.join(process_dir, folder_name)
            print(f"Processing {folder_name} for {split} set")
            
            syn_gt_data, syn_mag_data = process_path(folder_path, args.transform)
            
            # Save gt and mag to csv file
            save_dir = os.path.join('processed',transform , split, path_num, folder_name)
            os.makedirs(save_dir, exist_ok=True)
            
            gt_csv_path = os.path.join(save_dir, 'gt.csv')
            save_data_to_csv(syn_gt_data, gt_csv_path)
            
            mag_csv_path = os.path.join(save_dir, 'mag.csv')
            save_data_to_csv(syn_mag_data, mag_csv_path)
            print(f"Processing {save_dir} for {split} set successfully\n")

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="hkust4f", help="The root dir to process")
    parser.add_argument("--path_num", type=str, default="path2", help="The path to process")
    parser.add_argument("--transform", type=bool, default=False, help="Transform the magnetic signals into other forms")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Ratio of data to use for testing")

    args = parser.parse_args()
    main(args)
