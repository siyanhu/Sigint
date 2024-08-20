from datetime import datetime
import pandas as pd
import root_file_io as fio


def sort_paths(string_list):
    def extract_sort_key(path):
        (pathdir, pathname, pathext) = fio.get_filename_components(path)
        s = pathname
        parts = s.split('_')
        site_name = parts[0]
        floor_name = parts[1]
        date_time_str = '_'.join(parts[2:])
        date_time = datetime.strptime(date_time_str, '%Y-%m-%d_%H-%M-%S')
        
        # Return a tuple for sorting
        return (site_name, floor_name, date_time)

    # Sort the list using the extract_sort_key function
    return sorted(string_list, key=extract_sort_key)


class PathObject(object):
    def __init__(self, path_addr) -> None:
        __depthPath = fio.createPath(fio.sep, [path_addr], "Depth")
        if fio.file_exist(__depthPath):
            self.depthPath = __depthPath

        __rgbPath = fio.createPath(fio.sep, [path_addr], "RGB")
        if fio.file_exist(__rgbPath):
            self.rgbPath = __rgbPath

        __gtPath = fio.createPath(fio.sep, [path_addr], "Gt")
        if fio.file_exist(__gtPath):
            self.gtPath = __gtPath

        __sensorPath = fio.createPath(fio.sep, [path_addr], "Sensor")
        if fio.file_exist(__sensorPath):
            self.sensorPath = __sensorPath

        __gpsPath = fio.createPath(fio.sep, [path_addr], "GPS")
        if fio.file_exist(__gpsPath):
            self.gpsPath = __gpsPath

        __nerfPath = fio.createPath(fio.sep, [path_addr], "NeRF")
        if fio.file_exist(__nerfPath):
            self.nerfPath = __nerfPath

    def loadDepth(self):
        if self.depthPath:
            depth_files = fio.traverse_dir(self.depthPath, full_path=True, towards_sub=False)
            depth_files = fio.filter_ext(depth_files, filter_out_target=False, ext_set=fio.img_ext_set)
            return depth_files
        return []
    
    def loadRGB(self):
        if self.rgbPath:
            rgb_files = fio.traverse_dir(self.rgbPath, full_path=True, towards_sub=False)
            rgb_files = fio.filter_ext(rgb_files, filter_out_target=False, ext_set=fio.img_ext_set)
            return rgb_files
        return []
    
    def loadGPS(self):
        gps_data = []
        compass_data = []
        if self.gpsPath:
            gps_file_path = fio.createPath(fio.sep, [self.gpsPath], 'gps.txt')
            if fio.file_exist(gps_file_path):
                with open(gps_file_path, 'r') as file_gps:
                    for line in file_gps:
                        # Split the line into its components
                        timestamp, latitude, longitude, v_accuracy, h_accuracy = line.strip().split(',')
                        
                        # Create a dictionary for each line
                        row = {
                            'timestamp': float(timestamp),
                            'latitude': float(latitude),
                            'longitude': float(longitude),
                            'vertical_accuracy': float(v_accuracy),
                            'horizontal_accuracy': float(h_accuracy)
                        }
                        gps_data.append(row)

            compass_file_path = fio.createPath(fio.sep, [self.gpsPath], 'compass.txt')
            if fio.file_exist(compass_file_path):
                with open(compass_file_path, 'r') as file_comp:
                    for line in file_comp:
                        # Split the line into its components
                        timestamp, headingx, headingy, headingz = line.strip().split(',')
                        # Create a dictionary for each line
                        row = {
                            'timestamp': float(timestamp),
                            'x': float(headingx),
                            'y': float(headingy),
                            'z': float(headingz)
                        }
                        
                        compass_data.append(row)

        gps_df = pd.DataFrame()
        compass_df = pd.DataFrame()
        if len(gps_data):
            gps_df = pd.DataFrame(gps_data)
        if len(compass_data):
            compass_df = pd.DataFrame(compass_data)
        return (gps_df, compass_df)
    
    def loadGt(self):
        if self.gtPath:
            vis_path = fio.createPath(fio.sep, [self.gtPath], 'vis.txt')
