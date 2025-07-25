import numpy as np
import polars as pl
import torch

class AliExpressDataset(torch.utils.data.Dataset):
    """
    AliExpress Dataset
    This is a dataset gathered from real-world traffic logs of the search system in AliExpress
    Reference:
        https://tianchi.aliyun.com/dataset/dataDetail?dataId=74690
        Li, Pengcheng, et al. Improving multi-scenario learning to rank in e-commerce by exploiting task relationships in the label space. CIKM 2020.
    """

    def __init__(self, dataset_path, n_rows=None):
        if "test" in str(dataset_path):
            data = pl.read_csv(dataset_path, n_rows=n_rows).to_numpy()[500000:505000, 1:]
        else:
            data = pl.read_csv(dataset_path, n_rows=n_rows).to_numpy()[:500000, 1:]
        self.categorical_data = data[:, :16].astype(int)
        self.numerical_data = data[:, 16: -2].astype(np.float32)
        self.labels = data[:, -2:].astype(np.float32)
        self.numerical_num = self.numerical_data.shape[1]
        self.field_dims = np.max(self.categorical_data, axis=0) + 1

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        return self.categorical_data[index], self.numerical_data[index], self.labels[index]