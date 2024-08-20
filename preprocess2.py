import PathClass
from PathClass import fio, PathObject, sort_paths

if __name__ == '__main__':
    data_dir = fio.createPath(fio.sep, [fio.getParentDir(), 'G105'])
    path_dirs = fio.traverse_dir(data_dir, full_path=True, towards_sub=False)
    path_dirs = fio.filter_folder(path_dirs, filter_out=False, filter_text='DEFAULT_DEFAULT')
    sorted_path_dirs = sort_paths(path_dirs)
    print(sorted_path_dirs)