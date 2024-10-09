import os
import sys
sys.path.append(os.getcwd())
import pandas as pd
from glob2 import glob 
from natsort import natsorted
import h5py
import json
import numpy as np


# def raise_error_path_check(path):
#     if not os.path.exists(path):
#         raise FileNotFoundError('The file path is incorrect. Please double check.')
#
#
# def dir_check_make_new(path):
#     if not os.path.exists(path):
#         os.makedirs(path)


class DataReadWrite:

    def __init__(self):
        pass

    # ____________________________ GET SUB-/DIRECTORIES ____________________________

    def dir_check(self, dir: str) -> bool:
        """
        Check if path exists. Whether it is a file or a directory, if it exists it will return True.

        If the path exists, but is a file and not a directory,
        isdir will return False. Meanwhile, exists will return True in both cases.'

        ____________
        :param dir: string path to check.
        :return: True if path exists, otherwise raise FileNotFoundError.

        Example
        ______________

        from pathlib import Path
        x = dir # r'{dir}
        p = Path(x)

        rw = DataReadWrite()
        existss = rw.dir_check(p)
        """

        if os.path.exists(dir):
            return True
        else:
            raise FileNotFoundError(f'Cant find {dir}')

    def check_dir_exist(self, path):
        """
        Create a new directory if the given directory does not exist.
        :param path: directory to be checked
        """

        isExist = os.path.exists(path)
        if not isExist:
            #os.chmod(path, 0o777)
            os.makedirs(path)
            print("The new directory is created!")

    # def get_subdirs(path):
    #     subdirs = []
    #     with os.scandir(path) as entries:
    #         for entry in entries:
    #             if entry.is_dir():
    #                 subdirs.append(entry.name)
    #     return subdirs

    def get_subdirs(self, rootdir: str) -> list:
        """
        Function finds from the root directory all folders. Each folder corresponds to single subject.

        'If the path exists, but is a file and not a directory,
        isdir will return False. Meanwhile, exists will return True in both cases.'

        ______________
        :param rootdir: root directory with all subjects data
        :return: dirs: naturally sorted list of subjects folders' directories

        Example
        ______________
        from pathlib import Path
        x = rootdir # r'{rootdir}
        p = Path(x)

        rw = DataReadWrite()
        subdirectories_list = rw.get_subdirs(p)
        """

        self.dir_check(rootdir)
        children_dir = [os.path.join(rootdir, folder) for folder in os.listdir(rootdir)]
        dirs = natsorted(children_dir)
        return dirs

    def get_recursive_subdirs(self, rootpath: str, ext: str) -> list:
        """
        Find recursively all files with specified extension from root path. All subdirectories of subdirectories are
        searched through.

        __________
        :param rootpath: directory from which all files with specified extension will be found recursively.
        :param ext: extension of files you are looking at; example "*.pkl"
        :return: list of all the files in root directory and all subdirectories (recursively) with specified extension.

        Example
        ______________
        x = rootpath #r'{rootpath}
        rw = DataReadWrite()
        list_pkl_files = rw.search_files_subdirs(x, "*.pkl")

        """

        self.dir_check(rootpath)
        return glob(rootpath + "/**/" + str(ext), recursive=True)

    def search_files_subdir(self, path: str, ext: str) -> list:
        """
        List files only within the subdirectory of the path (one level down) with
        specified extension, example ".pkl", ".mat" etc.

        ___________
        :param path: path to directory where files of defined extension are listed.
        :return: list of files (absolute path) with extension (e.g. ".mat", ".pkl", ".csv")

        Example
        ______________
        from pathlib import Path
        x = path #rootdir # r'{rootdir}
        p = Path(x)
        rw = DataReadWrite()
        list_pkl_files = rw.search_files_subdir(p, ".pkl")
        """

        self.dir_check(path)
        files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(ext)]

        if len(files) == 0:
            print(f"Folder {path}  has no {ext} files. \nFolder includes:  {os.listdir(path)}.")

        return files

    @staticmethod
    def write_h5py(data_path: str, data):
        '''
        :param data_path: str  with *.hdf5 extension
        :param data: data to save
        '''
        hf = h5py.File("data.hdf5", "w")
        hf.create_dataset('data', data=data)
        hf.close()

    def read_h5py(self, data_path: str):
        """
        :param data_path: str  with *.hdf5 extension
        :return d: returns list of hdf5 objects.
        These should be later convert to numpy array's if needed
        """
        hf = h5py.File(data_path, 'r')
        d = []
        for k in hf.keys():
            data = hf.get(k)
            d.append(data)
        # hf.close()
        return d

    def write_npy(self, fname, data):
        """
        Write data to file with *.npy extension
        :param fname: str filename
        :param data: ndarray with data
        """
        with open(fname, 'wb') as f:
            np.save(fname, data)

    def read_npy(self, fname):
        """
        Read dxata from *.npy file
        :param fname: str file path with data
        :return: ndarray with data
        """
        self.dir_check(fname)
        with open(fname, 'rb') as f:
            data = np.load(f)
        return data

    def read_csv(self, fname):
        """
        Read file with *.csv extension
        :param fname: str file path with data
        :return: pandas dataframe with data
        """
        self.dir_check(fname)
        return pd.read_csv(fname)

    def read_file_list(self, dir, extension='.npy'):
        return [f for f in os.listdir(dir) if f.endswith(extension)]
