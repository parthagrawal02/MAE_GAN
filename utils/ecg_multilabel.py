import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import re
import wfdb
from wfdb import processing
import pdb
import pandas as pd

class CustomDataset(Dataset):

    def __init__(self, data_path: str = "", start: int = 0, end: int = 46):
        self.data_path = data_path
        self.data = []
        y = []
        if(sampling_rate == 100):
            filepath = [path + f for f in df.filename_lr]
        else:
            filepath = [path + f for f in df.filename_hr]

        self.data.append([filepath, output_array])

    def multihot_encoder(labels, n_categories = 1, dtype=torch.float32):
        label_set = set()
        for label_list in labels:
            label_set = label_set.union(set(label_list))
        label_set = sorted(label_set)

        multihot_vectors = []
        for label_list in labels:
            multihot_vectors.append([1 if x in label_list else 0 for x in label_set])
        if dtype is None:
            return pd.DataFrame(multihot_vectors, columns=label_set)
        return torch.Tensor(multihot_vectors).to(dtype)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ecg_path, class_name = self.data[idx]
        ecg_record = wfdb.rdsamp(ecg_path[:-4])
        lx = []
        for chan in range(ecg_record[0].shape[1]):
            resampled_x, _ = wfdb.processing.resample_sig(ecg_record[0][:, chan], 500, 100)
            lx.append(resampled_x)

        class_id = self.class_map[class_name]
        ecg_tensor = torch.from_numpy(np.array(lx))
        img_tensor = ecg_tensor[None, :, :]
        mean = img_tensor.mean(dim=-1, keepdim=True)
        var = img_tensor.var(dim=-1, keepdim=True)
        img_tensor = (img_tensor - mean) / (var + 1.e-6)**.5
        class_id = torch.tensor([class_id])
        return img_tensor, class_id
    


    def load_raw_data(df, sampling_rate, path):
        if(sampling_rate == 100):
            data = [path + f for f in df.filename_lr]
        else:
            data = [path + f for f in df.filename_hr]
        return data
    
data = np.array([signal for signal, meta in data])

    path = self.data_path
    sampling_rate = 100

    # load and convert annotation data
    Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Load raw signal data
    X = load_raw_data(Y, sampling_rate, path)

    # Load scp_statements.csv for diagnostic aggregation
    agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    # Apply diagnostic superclass
    Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

    # Split data into train and test
    test_fold = 10
    # Train
    X_train = X[np.where(Y.strat_fold != test_fold)]
    y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
    # Test
    X_test = X[np.where(Y.strat_fold == test_fold)]
    y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass

    def multihot_encoder(labels, n_categories = 1, dtype=torch.float32):
        label_set = set()
        for label_list in labels:
            label_set = label_set.union(set(label_list))
        label_set = sorted(label_set)

        multihot_vectors = []
        for label_list in labels:
            multihot_vectors.append([1 if x in label_list else 0 for x in label_set])
        if dtype is None:
            return pd.DataFrame(multihot_vectors, columns=label_set)
        return torch.Tensor(multihot_vectors).to(dtype)
    X_train = torch.tensor(X_train.transpose(0, 2, 1))
    mean = X_train.mean(dim=-1, keepdim=True)
    var = X_train.var(dim=-1, keepdim=True)
    X_train = (X_train - mean) / (var + 1.e-6)**.5
    X_test = torch.tensor(X_test.transpose(0, 2, 1))
    mean = X_test.mean(dim=-1, keepdim=True)
    var = X_test.var(dim=-1, keepdim=True)
    X_test = (X_test - mean) / (var + 1.e-6)**.5

    y_train = multihot_encoder(y_train, n_categories = 5)
    y_test = multihot_encoder(y_test, n_categories= 5)
