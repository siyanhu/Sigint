import os
import csv
from datetime import datetime, timezone, timedelta
import math
import numpy as np
from operator import itemgetter
import sys
import random

threshold_ts = 10 #ms

def time2ts(time_str):
  hk_time = datetime.strptime(time_str, "%Y-%m-%d_%H-%M-%S")
  # Define the HK timezone (UTC+8)
  hk_timezone = timezone(timedelta(hours=8))
  hk_time = hk_time.replace(tzinfo=hk_timezone)
  timestamp_ms = int(hk_time.timestamp() * 1000)
  return timestamp_ms


def ts2time(ts):
  # Convert milliseconds to seconds
  timestamp_sec = ts / 1000
  # Convert to datetime in UTC
  dt = datetime.fromtimestamp(timestamp_sec, tz=timezone.utc)
  hk_timezone = timezone(timedelta(hours=8))

  hk_time = dt.astimezone(hk_timezone)
  readable_hk_date = hk_time.strftime('%Y-%m-%d %H:%M:%S %Z')
  return readable_hk_date


def trunc_name(img_name):
  return img_name[4:-4]


def compute_start_end(dir, start_entry = None, end_entry = None, return_ts = True):
  # Specify the folder path
  rgb_dir = os.path.join(dir, 'RGB')

  # Get all entries in the folder and sort them by name
  entries = sorted(os.listdir(rgb_dir))

  if len(entries) > 1:
      
      start_entry = entries[0] if start_entry is None else start_entry      
      end_entry = entries[-1] if end_entry is None else end_entry

      truncated_first = trunc_name(start_entry)
      truncated_end = trunc_name(end_entry)
      print(f"Time of this path: {truncated_first} to {truncated_end}")

      if return_ts:
        return time2ts(truncated_first), time2ts(truncated_end)
      else:
        return truncated_first, truncated_end

  else:
      print("Not having enough rgb data!")


def process_gt_file(gt_csv_path, begin_ts, end_ts, negate_x=False, negate_y=True, negate_z=True):
    path_data = {'ts': [], 'x': [], 'y': [], "z": []}

    
    data = np.genfromtxt(gt_csv_path, delimiter=',', dtype=None, encoding=None, invalid_raise = False)
    

    x_factor = -1 if negate_x else 1
    y_factor = -1 if negate_y else 1
    z_factor = -1 if negate_z else 1

    prev_ts = None
    cumulative_idx = 0

    for row in data:

        try:
            current_ts = float(row[0])
            
            if prev_ts is not None:
                idx = current_ts - prev_ts
                cumulative_idx += idx

                if idx == 0:
                  print(f"Skipping invalid row: ts is {row[0]}")
                  continue
            else:
                idx = 0

            

            ts = int(cumulative_idx * 1000 + begin_ts)
            
            if ts > end_ts:
                break

            path_data['ts'].append(ts)
            path_data['x'].append(float(row[2]) * x_factor)
            path_data['y'].append(float(row[3]) * y_factor)
            path_data['z'].append(float(row[4]) * z_factor)

            prev_ts = current_ts

        except (ValueError, IndexError) as e:
            print(f"Error processing row {row}: {e}")
            continue

    return path_data


#For a ts in gt_data, find its closet timestamp in full_mag_data 
def find_closest_number_index(sequence, x): #sequence is a list of ts

    # Initialize the closest number, the difference, and the index
    closest = sequence[0]
    diff = abs(sequence[0] - x)
    closest_index = 0

    # Iterate through the sequence
    for i, num in enumerate(sequence):
        # Calculate the absolute difference between the current number and x
        curr_diff = abs(num - x)

        # If the current difference is smaller, update the closest number, difference, and index
        if curr_diff < diff:
            closest = num
            diff = curr_diff
            closest_index = i

        if curr_diff > diff:
            break

    return closest_index


def calMagFeature(mag, grav):
    magnitude = math.sqrt(sum(component**2 for component in grav))
    grav_norm = [component / magnitude for component in grav]
    dot_product = sum(component1 * component2 for component1, component2 in zip(mag, grav_norm))
    mag_along_grav = [component * dot_product for component in grav_norm]
    mag_orth_grav = [component1 - component2 for component1, component2 in zip(mag, mag_along_grav)]
    magnitide_along_grav = math.sqrt(sum(component**2 for component in mag_along_grav))
    if dot_product<0:
        magnitide_along_grav = -magnitide_along_grav
    magnitide_orth_grav = math.sqrt(sum(component**2 for component in mag_orth_grav))
    return [magnitide_along_grav,magnitide_orth_grav,math.sqrt(sum(component**2 for component in mag))]


def process_imu_file(imu_csv_path, transform):
    mag_data_transform = {'ts':[],'Bv':[], 'Bh':[], "Bp":[]}
    mag_data_raw = {'ts':[],'Bx':[], 'By':[], "Bz":[]}

    data = np.genfromtxt(imu_csv_path, delimiter=',', dtype=None, encoding=None, invalid_raise = False).tolist()

    # Sort data based on timestamp (first column)
    sorted_data = sorted(data, key=itemgetter(0))

    if transform:
      for row in sorted_data:
          
        mag = [float(row[-3]),float(row[-2]),float(row[-1])]
        grav = [float(row[-9]),float(row[-8]),float(row[-7])]
        
        Bv, Bh, Bp = calMagFeature(mag, grav)

        mag_data_transform["ts"].append(int(row[0]))
        mag_data_transform["Bv"].append(Bv)
        mag_data_transform["Bh"].append(Bh)
        mag_data_transform["Bp"].append(Bp)

      return mag_data_transform

    else:
      for row in sorted_data:


        Bx,By,Bz = [float(row[-3]),float(row[-2]),float(row[-1])]
        
        mag_data_raw["ts"].append(int(row[0]))
        mag_data_raw["Bx"].append(Bx)
        mag_data_raw["By"].append(By)
        mag_data_raw["Bz"].append(Bz)

      return mag_data_raw
      

#We assume a constant speed within 1 sec
#Generate gt for every imu data points that does not have a matched ts with ts in gt_data
def generate_gt(gt_data, full_mag_data):

  #for each ts in gt_data, find its closest ts in full_mag_data, store them in matched_idxes
  matched_idxes = []
  for gt_ts in gt_data['ts']:
    idx = find_closest_number_index(full_mag_data['ts'], gt_ts)
    matched_idxes.append(idx)

  gt_keys = list(gt_data.keys())
  mag_keys = list(full_mag_data.keys())
  syn_gt_data = {gtk: [] for gtk in gt_keys}
  syn_mag_data = {magk: [] for magk in mag_keys}

  for k in range(len(matched_idxes)):
    
    ts_k = full_mag_data['ts'][matched_idxes[k]]
    syn_gt_data['ts'].append(ts_k)
    syn_mag_data['ts'].append(ts_k)

    for i in range(1, 4):
      syn_mag_data[mag_keys[i]].append(full_mag_data[mag_keys[i]][matched_idxes[k]])
      syn_gt_data[gt_keys[i]].append(gt_data[gt_keys[i]][k])

    if k == len(matched_idxes)-1:
      break

    interval = matched_idxes[k+1] - matched_idxes[k] #how many points in between 2 matched ts


    if interval == 0:
      print(k)
      print(matched_idxes[k+1], matched_idxes[k])
      return matched_idxes, None, None

    #the avg distances in x,y,z directions between every 2 adjacent points in the interval
    diff = [] #diff in x, y, z directions
    for i in range(3):
      diff.append((gt_data[gt_keys[i]][k+1] - gt_data[gt_keys[i]][k])/interval) 


    #loop through every points in the interval, generate gt for each point
    for idx in range(matched_idxes[k]+1, matched_idxes[k+1]):
      
      syn_ts = full_mag_data['ts'][idx]
      syn_gt_data['ts'].append(syn_ts)
      syn_mag_data['ts'].append(syn_ts)

      for j in range(1,4):
        syn_mag_data[mag_keys[j]].append(full_mag_data[mag_keys[j]][idx])
        syn_gt_data[gt_keys[j]].append(syn_gt_data[gt_keys[j]][-1] + diff[j-1])

  return matched_idxes, syn_gt_data, syn_mag_data



def time_difference(timestamp1, timestamp2):
    dt1 = datetime.fromtimestamp(timestamp1/1000)
    dt2 = datetime.fromtimestamp(timestamp2/1000)
    return dt2 - dt1
  
def output_check(matched_idxes, full_mag_data, gt_data, syn_gt_data, syn_mag_data, threshold_ts = threshold_ts):
  
  for k in range(len(matched_idxes)):
    diff = full_mag_data['ts'][matched_idxes[k]] - gt_data['ts'][k]
    if abs(diff) > threshold_ts:
        error_message = f"Error: At index {k}, the diff between gt_ts and matched mag_ts is too large: {diff}ms"
        print(error_message)
        sys.exit(1) 
    
  length = matched_idxes[-1] - matched_idxes[0] + 1 #the length of syn_mag_data
  if not (len(syn_gt_data['ts']) == len(syn_mag_data['ts'])) or not (length == len(syn_gt_data['ts'])):
    print(f"Error: Length Not Match for syn_gt_data and syn_mag_data")
    sys.exit(1)
  
  mag_keys = list(full_mag_data.keys())
  i = random.randint(0, len(matched_idxes) - 1) #between 0 and len(matched_idxes) - 1
  if not (full_mag_data[mag_keys[2]][matched_idxes[i]] == syn_mag_data[mag_keys[2]][matched_idxes[i] - matched_idxes[0]]):
    print(f"Error: Mag Not Match for full_mag_data and syn_mag_data: Row {matched_idxes[i]}")
    sys.exit(1)
    
  print(f"Syn data length: {length}")
  print(f"Time diff in sec: {time_difference(syn_gt_data['ts'][0],syn_gt_data['ts'][-1]).total_seconds():.4f}s")



def read_tap(tap_file_path):
  data = np.loadtxt(tap_file_path, delimiter=',', ndmin=2)

  tap_data = {
      'ts': data[:, 0].tolist(),
      'x': data[:, 1].tolist(),
      'y': data[:, 2].tolist()
  }

  return tap_data


def read_all_tap(all_dirs):
    all_tap_data = []
    for folder_dir in all_dirs:
        tap_file_path = os.path.join(folder_dir, "Gt", "taps.txt")

        try:
            tap_data = read_tap(tap_file_path)
            all_tap_data.append(tap_data)
            print(f"Processed {folder_dir}: {len(tap_data['ts'])} taps")

        except Exception as e:
            print(f"Error processing {folder_dir}: {str(e)}")

    print(f"Total folders processed: {len(all_tap_data)}")
    return all_tap_data