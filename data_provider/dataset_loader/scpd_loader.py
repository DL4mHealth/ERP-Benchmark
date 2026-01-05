import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from data_provider.uea import (
    normalize_batch_ts,
    bandpass_filter_func,
    load_data_by_ids,
)
import warnings
import random

warnings.filterwarnings('ignore')


def get_id_list_scpd(args, label_path, a=0.6, b=0.8):
    '''
    Loads subject IDs for all, training, validation, and test sets for SCPD data
    Use healthy and Parkinson's disease subjects
    Args:
        args: arguments
        label_path: directory of label files
        a: ratio of ids in training set
        b: ratio of ids in training and validation set
    Returns:
        all_ids: list of all IDs
        train_ids: list of IDs for training set
        val_ids: list of IDs for validation set
        test_ids: list of IDs for test set
    '''
    # random shuffle to break the potential influence of human named ID order,
    # e.g., put all healthy subjects first or put subjects with more samples first, etc.
    # (which could cause data imbalance in training, validation, and test sets)
    data_list = []
    for filename in os.listdir(label_path):
        sub_label_path = os.path.join(label_path, filename)
        subject_label = np.load(sub_label_path)
        # [task_id, accuracy, subject_id, disease_id]
        # [subject_id, disease_id] For SCPD, subject ID and disease ID should be the same for all samples a subject
        subject_id_disease_id = subject_label[0, 2:4]
        data_list.append(subject_id_disease_id.reshape(1, -1))
    data_list = np.concatenate(data_list, axis=0)
    all_ids = list(data_list[:, 0])  # all subjects
    hc_list = list(data_list[np.where(data_list[:, 1] == 0)][:, 0])  # healthy IDs
    pd_list = list(data_list[np.where(data_list[:, 1] == 1)][:, 0])  # Parkinson's disease IDs
    if args.cross_val == 'fixed' or args.cross_val == 'mccv':  # fixed split or Monte Carlo cross-validation
        if args.cross_val == 'fixed':
            random.seed(42)  # fixed seed for fixed split
        else:
            random.seed(args.seed)  # random seed for Monte Carlo cross-validation

        random.shuffle(hc_list)
        random.shuffle(pd_list)

        train_ids = hc_list[:int(a * len(hc_list))] + pd_list[:int(a * len(pd_list))]
        val_ids = (hc_list[int(a * len(hc_list)):int(b * len(hc_list))] +
                   pd_list[int(a * len(pd_list)):int(b * len(pd_list))])
        test_ids = hc_list[int(b * len(hc_list)):] + pd_list[int(b * len(pd_list)):]

        return sorted(all_ids), sorted(train_ids), sorted(val_ids), sorted(test_ids)

    elif args.cross_val == 'loso':  # leave-one-subject-out cross-validation
        # take subject ID with index (args.seed-41) % len(all_ids) as test set, random seed start from 41
        all_ids = sorted(all_ids)
        test_ids = [all_ids[(args.seed - 41) % len(all_ids)]]
        train_ids = [id for id in all_ids if id not in test_ids]
        # randomly take 10% of the training set as validation set
        random.seed(args.seed)
        random.shuffle(train_ids)
        val_ids = train_ids

        return sorted(all_ids), sorted(train_ids), sorted(val_ids), sorted(test_ids)
    else:
        raise ValueError('Invalid cross_val. Please use fixed, mccv, or loso.')


class SCPDLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.no_normalize = args.no_normalize
        self.root_path = root_path
        self.data_path = os.path.join(root_path, 'Feature/')
        self.label_path = os.path.join(root_path, 'Label/')

        a, b = 0.6, 0.8
        self.all_ids, self.train_ids, self.val_ids, self.test_ids = get_id_list_scpd(args, self.label_path, a, b)

        if flag == 'TRAIN':
            ids = self.train_ids
            print('train ids:', ids)
        elif flag == 'VAL':
            ids = self.val_ids
            print('val ids:', ids)
        elif flag == 'TEST':
            ids = self.test_ids
            print('test ids:', ids)
        elif flag == 'PRETRAIN':
            ids = self.all_ids
            print('all ids:', ids)
        else:
            raise ValueError('Invalid flag. Please use TRAIN, VAL, TEST, or PRETRAIN.')

        self.X, self.y = load_data_by_ids(self.data_path, self.label_path, ids, args)
        self.X = normalize_batch_ts(self.X)

        self.y = self.y[:, [3, 2]]  # only keep [disease_id, subject_id] for SCPD
        """self.y = self.y[:, [1, 2]]  # only keep [accuracy, subject_id] for SCPD
        mask = (self.y[:, 0] != 99)  # remove all y = 99 index
        self.X = self.X[mask]
        self.y = self.y[mask]"""

        self.max_seq_len = self.X.shape[1]

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), \
               torch.from_numpy(np.asarray(self.y[index]))

    def __len__(self):
        return len(self.y)
