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
import matplotlib.pyplot as plt


from helper_function import \
    compute_start_end, \
    process_gt_file, \
    process_imu_file, \
    generate_gt, \
    output_check, \
    read_tap    

threshold_ts = 100
PIXEL_TO_METER_SCALE = 13.913

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
    
    #Synchronize gts
    
    
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

#THIS only apply to data with more than 1 taps!
#Here we use translation.z and translation.y as x and y
def local2gloabl(folder_dir, gt_data):
  print(f"Processing {folder_dir}")

  #1. read taps.txt
  tap_file_path = os.path.join(folder_dir, "Gt", "taps.txt")
  tap_data = read_tap(tap_file_path)
  
  #Computer the direction
  scale = 1 if tap_data['x'][-1] - tap_data['x'][0] > 0 else -1

  #Get the global start position in meter
  x_local2global = tap_data['x'][0] / PIXEL_TO_METER_SCALE
  y_local2global = tap_data['y'][0] / PIXEL_TO_METER_SCALE
  print(f"Start: {x_local2global, y_local2global}")
  print(f"End: {tap_data['x'][-1] / PIXEL_TO_METER_SCALE, tap_data['y'][-1] / PIXEL_TO_METER_SCALE } ")
  
  #convert gt 
  gt_keys = list(gt_data.keys())
  gt_data_global = {k : [] for k in gt_keys[:-1]}
  for ts, x, y in zip(gt_data['ts'], gt_data['z'], gt_data['y']):
    gt_data_global['ts'].append(ts)
    gt_data_global['x'].append(x_local2global + x * scale)
    gt_data_global['y'].append(y_local2global + y)

  return tap_data, gt_data_global



def plot_multiple_paths(all_gt_global, labels= None, equal=False):
    fig, ax = plt.subplots(figsize=(10, 3))

    starts = [0 for i in range(len(all_gt_global))]
    ends = [len(gt['ts']) for gt in all_gt_global]
    x_list = [all_gt_global[i]['x'] for i in range(len(all_gt_global))]
    y_list = [all_gt_global[i]['y'] for i in range(len(all_gt_global))]

    if labels is None:
        labels = [f'Path {i+1}' for i in range(len(x_list))]

    colors = plt.cm.rainbow(np.linspace(0, 1, len(x_list)))

    for i, (x, y, start, end, label, color) in enumerate(zip(x_list, y_list, starts, ends, labels, colors)):
        # Plot the path
        ax.plot(x[start:end], y[start:end], '-o', markersize=4, label=label, color=color)
        
        # Plot start and end points
        ax.plot(x[start], y[start], 'go', markersize=10)
        ax.plot(x[end-1], y[end-1], 'ro', markersize=10)

    if equal:
        ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x coordinate')
    ax.set_ylabel('y coordinate')
    ax.set_title('Multiple Paths')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()
    

def main(args):    
    root_dir = args.root_dir
    path_num = args.path_num
    process_dir = os.path.join(root_dir, path_num)
    
    transform_yes = bool(args.transform)
    transform = 'transformed' if transform_yes else 'no_transform'

    # Perform train-test split
    train_folders, test_folders = train_test_split(process_dir, args.test_ratio)
    
    all_gt_global = []

    for split, folders in [('train', train_folders), ('test', test_folders)]:
        for folder_name in folders:
            folder_path = os.path.join(process_dir, folder_name)
            print(f"Processing {folder_name} for {split} set")
            
            syn_gt_data, syn_mag_data = process_path(folder_path, args.transform)
            
            #convert from local to global frame
            tap_data, gt_data_global = local2gloabl(folder_path, syn_gt_data)
            all_gt_global.append(gt_data_global)
            
            # Save gt and mag to csv file
            save_dir = os.path.join('processed', transform , path_num, split, folder_name) #change
            print(f"Save to {save_dir}")
            os.makedirs(save_dir, exist_ok=True)
            
            gt_csv_path = os.path.join(save_dir, 'gt.csv')
            save_data_to_csv(gt_data_global, gt_csv_path)
            
            mag_csv_path = os.path.join(save_dir, 'mag.csv')
            save_data_to_csv(syn_mag_data, mag_csv_path)
            print(f"Processing {save_dir} for {split} set successfully\n")
            
    plot_multiple_paths(all_gt_global, equal = False)

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="hkust4f", help="The root dir to process")
    parser.add_argument("--path_num", type=str, default="new", help="The path to process")
    parser.add_argument("--transform", type=bool, default=False, help="Transform the magnetic signals into other forms")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Ratio of data to use for testing")

    args = parser.parse_args()
    main(args)
