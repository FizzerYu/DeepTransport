import os, random
import numpy as np
import os
import pandas as pd
import datetime as dtm
import pickle
import numpy as np
import subprocess
from tqdm import tqdm
import logging
logging.basicConfig(
    format=
    "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
    level=logging.INFO)
logger = logging.getLogger(__name__)

def set_global_seed(seed=42, more="torch"):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed) 
    if more == "torch":
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic=True
    elif more == "tf":
        import tensorflow as tf
        tf.random.set_seed(seed)
    logger.info(f'Set Seed={seed} Successfully!')

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
        https://www.kaggle.com/gemartin/load-data-reduce-memory-usage?scriptVersionId=3684066&cellId=3       
    """
    start_mem = df.memory_usage().sum() / 1024**2
    logger.info('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    pass_cols = []
    for col in tqdm(df.columns):
        col_type = df[col].dtype.name
        
        if col_type.startswith("int"):
            c_min = df[col].min()
            c_max = df[col].max()
            if col_type == 'int8':
                continue
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df.loc[:, col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df.loc[:, col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df.loc[:, col] = df[col].astype(np.int32)
            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                df.loc[:, col] = df[col].astype(np.int64)  
        elif col_type.startswith('float'):
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type) == 'float8':
                continue
            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                df.loc[:, col] = df[col].astype(np.float16)
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df.loc[:, col] = df[col].astype(np.float32)
            else:
                df.loc[:, col] = df[col].astype(np.float64)
        else:
            pass_cols.append(col)

    logger.info("pass columns: ", pass_cols)
    end_mem = df.memory_usage().sum() / 1024**2
    logger.info('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    logger.info('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


def reduce_numpy_mem_usage(array):
    int64min, int64max = np.iinfo(np.int64).min, np.iinfo(np.int64).max
    int32min, int32max = np.iinfo(np.int32).min, np.iinfo(np.int32).max
    int16min, int16max = np.iinfo(np.int16).min, np.iinfo(np.int16).max
    int8min, int8max = np.iinfo(np.int8).min, np.iinfo(np.int8).max
    _max, _min = np.max(array), np.min(array)
    origin_dtype = array.dtype.name
    if origin_dtype.startswith("int"):
        if _min > int8min and _max < int8max:
            array = array.astype("int8")
            after_dtype = "int8"
        elif _min > int16min and _max < int16max:
            array = array.astype("int16")
            after_dtype = "int16"
        elif _min > int32min and _max < int32max:
            array = array.astype("int32")
            after_dtype = "int32"
        elif _min > int64min and _max < int64max:
            array = array.astype("int64")
            after_dtype = "int64"
        else:
            raise Exception
    logger.info(f"\t\tSqueeze array from {origin_dtype} to {after_dtype}")
    return array
        
###################################################################################################################
# save files
def bash2py(shell_command, split=True):
    """
    Args:
        shell_command: str, shell_command
        No capture_output arg for < py3.7 !
    example:
        bash2py('du -sh')    
    """
    res = subprocess.run(shell_command,
                 shell=True,
                 stdout=subprocess.PIPE,
                 stderr=subprocess.PIPE)
    if res.returncode == 0:
        logger.info(f"Execute <{shell_command}> successfully!")
    else:
        raise Exception(f"ERROR: {res.stderr.decode('utf-8')}")
    res = res.stdout.decode('utf-8').strip()  #.split('\n')
    if split:
        return res.split('\n')
    else:
        return res
                        
def mkdirs(dir2make):
    if isinstance(dir2make, list):
        for i_dir in dir2make:
            if not os.path.exists(i_dir):
                os.makedirs(i_dir)
    elif isinstance(dir2make, str):
        if not os.path.exists(dir2make):
            os.makedirs(dir2make)
    else:
        raise ValueError("dir2make should be string or list type.")

        
BASE_DIR_cwd = os.getcwd()
BASE_DIR_abs = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
BASE_DIR = BASE_DIR_cwd
        
class pkl_tool():
    @staticmethod
    def savepkl(data, nm_marker=None, prefix='CACHE_', postfix='.pkl', dt_format='%Y%m%d_%Hh'):
        mkdirs(os.path.join(BASE_DIR, 'cache'))
        name_ = dtm.datetime.now().strftime(dt_format)
        if nm_marker is not None:
            name_ = nm_marker
        path_ = os.path.join(BASE_DIR, f'cache/CACHE_{name_}.pkl')
        with open(path_, 'wb') as file:
            pickle.dump(data, file, protocol=4)
        logger.info(f'Cache Successfully! File name: {path_}')

    @staticmethod
    def readpkl(file_nm,
             pure_nm=False,
             base_dir=None,
             prefix='CACHE_',
             postfix='.pkl'):
        if pure_nm:
            file_nm = prefix + file_nm + postfix
        if base_dir is None:
            base_dir = os.path.join(BASE_DIR, 'cache')
        file_path = os.path.join(base_dir, file_nm)
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        logger.info(f'Successfully Reload: {file_path}')
        return data

    @staticmethod
    def delallpkl(AreYouSure):
        if AreYouSure == 'clear_cache':
            bash2py(f"cd {BASE_DIR} && rm -rf cached_data/")
        else:
            logger.info(
                "If you truely wanna clear all cached data, set AreYouSure as 'clear_cache'"
            )