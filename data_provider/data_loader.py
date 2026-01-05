import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from data_provider.uea import (
    normalize_batch_ts,
    bandpass_filter_func,
)
from utils.tools import get_channel_index
import warnings
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from scipy.signal import resample

from data_provider.dataset_loader.pd_sim_loader import PDSIMLoader
from data_provider.dataset_loader.pd_odd_loader import PDODDLoader
from data_provider.dataset_loader.cesca_aodd_loader import CESCAAODDLoader
from data_provider.dataset_loader.cesca_vodd_loader import CESCAVODDLoader
from data_provider.dataset_loader.cesca_flanker_loader import CESCAFLANKERLoader
from data_provider.dataset_loader.cognision_erp_loader import CognisionERPLoader
from data_provider.dataset_loader.mtbi_odd_loader import MTBIODDLoader
from data_provider.dataset_loader.scpd_loader import SCPDLoader
from data_provider.dataset_loader.rlpd_loader import RLPDLoader
from data_provider.dataset_loader.aopd_loader import AOPDLoader
from data_provider.dataset_loader.adhd_wmri_loader import ADHDWMRILoader
from data_provider.dataset_loader.nserp_msit_loader import NSERPMSITLoader
from data_provider.dataset_loader.nserp_odd_loader import NSERPODDLoader

# data folder dict to loader mapping
data_folder_dict = {
    # should use the same name as the dataset folder
    'PD-SIM': PDSIMLoader,
    'PD-ODD': PDODDLoader,
    'CESCA-AODD': CESCAAODDLoader,
    'CESCA-VODD': CESCAVODDLoader,
    'CESCA-FLANKER': CESCAFLANKERLoader,
    'Cognision-ERP': CognisionERPLoader,
    'mTBI-ODD': MTBIODDLoader,
    'SCPD': SCPDLoader,
    'RLPD': RLPDLoader,
    'AOPD': AOPDLoader,
    'ADHD-WMRI': ADHDWMRILoader,
    'NSERP-MSIT': NSERPMSITLoader,
    'NSERP-ODD': NSERPODDLoader,
}
warnings.filterwarnings('ignore')


class MultiDatasetsLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.no_normalize = args.no_normalize
        self.root_path = root_path

        print(f"Loading {flag} samples from multiple datasets...")
        if flag == 'PRETRAIN':
            data_folder_list = args.pretraining_datasets.split(",")
        elif flag == 'TRAIN':
            data_folder_list = args.training_datasets.split(",")
        elif flag == 'TEST' or flag == 'VAL':
            data_folder_list = args.testing_datasets.split(",")
        else:
            raise ValueError("flag must be PRETRAIN, TRAIN, VAL, or TEST")
        print(f"Datasets used ", data_folder_list)
        self.X, self.y = None, None
        global_ids_range = 1  # count global subject number to avoid duplicate IDs in multiple datasets
        for i, data in enumerate(data_folder_list):
            if data not in data_folder_dict.keys():
                raise Exception("Data not matched, "
                                "please check if the data folder name in data_folder_dict.")
            else:
                Data = data_folder_dict[data]
                data_set = Data(
                    root_path=os.path.join(args.root_path, data),
                    args=args,
                    flag=flag,
                )
                # add dataset ID to the third column of y, id starts from 1
                data_set.y = np.concatenate((data_set.y, np.full(data_set.y[:, 0].shape, i + 1).reshape(-1, 1)), axis=1)
                print(f"{data} data shape: {data_set.X.shape}, {data_set.y.shape}")
                if self.X is None or self.y is None:
                    self.X, self.y = data_set.X, data_set.y
                    global_ids_range = max(len(data_set.all_ids), max(data_set.all_ids))
                else:
                    # number of subjects or max subject ID in the current dataset
                    current_ids_range = max(len(data_set.all_ids), max(data_set.all_ids))
                    # update subject IDs in the current dataset by adding global_ids_range
                    data_set.y[:, 1] += global_ids_range
                    # update global subject number
                    global_ids_range += current_ids_range
                    # concatenate data from different datasets
                    self.X, self.y = (np.concatenate((self.X, data_set.X), axis=0),
                                      np.concatenate((self.y, data_set.y), axis=0))

        self.X, self.y = shuffle(self.X, self.y, random_state=42)
        self.max_seq_len = self.X.shape[1]
        # print(f"Unique subjects used in {flag}: ", len(np.unique(self.y[:, 1])))
        print()

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), \
               torch.from_numpy(np.asarray(self.y[index]))

    def __len__(self):
        return len(self.y)
