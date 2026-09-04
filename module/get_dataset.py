from torch_geometric.data import Data, Dataset
import numpy as np
import torch
import os
import shutil
from tqdm import tqdm

class TrajectoryDataset(Dataset):
    def __init__(self, root, file_path, transform=None, force_reload=False):
        """
        force_reload=True: force-delete the processed directory and re-process
        """
        self.file_path = file_path


        if force_reload and os.path.exists(os.path.join(root, 'processed')):
            print("Force-clearing old cache...")
            shutil.rmtree(os.path.join(root, 'processed'))


        self.raw_data_cache = self._load_raw_data()


        super().__init__(root, transform)

    def _load_raw_data(self):
        """Load the raw NPZ data into memory"""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Raw NPZ file not found: {self.file_path}")

        print(f"Loading raw data from {self.file_path} into memory...")
        with np.load(self.file_path, allow_pickle=True) as dat:

            cache = {
                'atoms': dat['atoms'],
                'pos': dat['pos'],
                'forces': dat['forces'],
                'energy': dat['energy'],
                'vbias': dat['vbias'],
                'cell': dat['cell'],
            }
        return cache

    @property
    def num_frames(self):
        return len(self.raw_data_cache["energy"])

    def raw_file_names(self):
        return [os.path.basename(self.file_path)]

    def processed_file_names(self):
        return ['processing_complete.tag']

    def download(self):
        pass

    def process(self):
        print("First run (or cache cleared): processing and saving data as .pt files...")


        os.makedirs(self.processed_dir, exist_ok=True)

        for i in tqdm(range(self.num_frames), desc="Processing frames"):


            data = Data(
                x=torch.tensor(np.array(self.raw_data_cache['atoms'][i], dtype=np.int64), dtype=torch.int64),
                pos=torch.tensor(np.array(self.raw_data_cache['pos'][i], dtype=np.float64), dtype=torch.float),
                force_target=torch.tensor(np.array(self.raw_data_cache['forces'][i], dtype=np.float64), dtype=torch.float),
                energy_target=torch.tensor(np.array(self.raw_data_cache['energy'][i], dtype=np.float64), dtype=torch.float),
                vbias=torch.tensor(np.array(self.raw_data_cache['vbias'][i], dtype=np.float64), dtype=torch.float),
                cell=torch.tensor(np.array(self.raw_data_cache['cell'][i], dtype=np.float64), dtype=torch.float)
            )
            torch.save(data, os.path.join(self.processed_dir, f"data_{i}.pt"))

        with open(os.path.join(self.processed_dir, 'processing_complete.tag'), 'w') as f:
            f.write(str(self.num_frames))

    def len(self):
        return self.num_frames

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f"data_{idx}.pt"), weights_only=False)
        return data
