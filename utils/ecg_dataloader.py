import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import re
import wfdb
from wfdb import processing
import pdb
class CustomDataset(Dataset):
    def __init__(self, data_path: str = "", start: int = 0, end: int = 46):
        self.class_map  = {
        426177001: 1,
        426783006: 2,
        164889003: 3,
        427084000: 4,
        164890007: 5,
        427393009: 6,
        426761007: 7,
        713422000: 8,
        233896004: 9,
        233897008: 0
        }
        self.data_path = data_path
        self.data = []
        y = []
        for n in range(start, end):
            for j in range(0, 10):
                for filepath in glob.iglob(self.data_path + '/WFDBRecords/' + f"{n:02}" +  '/' + f"{n:02}" + str(j) +  '/*.hea'):
                    try:
                        ecg_record = wfdb.rdsamp(filepath[:-4])
                    except Exception:
                        continue
                    # pdb.set_trace()
                    if(np.isnan(ecg_record[0]).any()):
                        print(str(filepath))
                        continue
                    numbers = re.findall(r'\d+', ecg_record[1]['comments'][2])
                    output_array = list(map(int, numbers))
                    for j in output_array: # Only classify into one of the predecided classes.
                        if int(j) in self.class_map:
                            output_array = j
                    if isinstance(output_array, list):
                        continue
                    y.append(output_array)
                    self.data.append([filepath, output_array])
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