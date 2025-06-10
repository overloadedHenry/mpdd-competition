import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
import os

class MyDataset(Dataset):
    def __init__(self, root_path, json_data, ps_feature_path, task_label_num, feature_max_len):

        self.json_data = json_data
        self.task_label_num = task_label_num
        self.root_path = Path(root_path)
        self.ps_feature_path = ps_feature_path
        self.task_label_num = task_label_num
        self.ps_feature = self.load_personalized_feature(self.ps_feature_path)
        self.feature_max_len = feature_max_len
        # self.ps_feature = torch.tensor(self.ps_feature, dtype=torch.float32)

    def __getitem__(self, index):

        entry = self.json_data[index]
        audio_path = Path(entry["audio_feature_path"])
        vision_path = Path(entry["video_feature_path"])
        audio_feature = np.load(self.root_path / Path("1s/Audio/wav2vec") /audio_path, allow_pickle=True)
        audio_feature = torch.tensor(audio_feature, dtype=torch.float32)
        vision_feature = np.load(self.root_path/Path("1s/Visual/openface") /vision_path, allow_pickle=True)
        vision_feature = torch.tensor(vision_feature, dtype=torch.float32)
        audio_feature = self.pad_or_truncate(audio_feature, self.feature_max_len)
        vision_feature = self.pad_or_truncate(vision_feature, self.feature_max_len)
        # ps_feature = torch.tensor(ps_feature, dtype=float32)
        task_label = None
        if self.task_label_num == 2:

            task_label = self.json_data[index]["bin_category"]
        elif self.task_label_num == 3:
            task_label = self.json_data[index]["tri_category"]
        elif self.task_label_num == 5:
            task_label = self.json_data[index]["pen_category"]


        filepath = entry['audio_feature_path']  # the filename containing path to features
        filename = os.path.basename(filepath)
        # Extract person ids and convert to integers
        person_id = int(filename.split('_')[0])
        personalized_id = str(person_id)

        if personalized_id in self.ps_feature:
            personalized_feature = torch.tensor(self.ps_feature[personalized_id], dtype=torch.float32)
            
        else:
            # If no personalized feature found, use a zero vector
            personalized_feature = torch.zeros(1024, dtype=torch.float32)
            print(f"❗Personalized feature not found for id: {personalized_id}")
             
        if personalized_feature.shape[0] == 1024:
            personalized_feature = personalized_feature.unsqueeze(0)
        return {
            "a_feature": audio_feature,
            "v_feature": vision_feature,
            "p_feature": personalized_feature,
            'task_label': task_label
        }

    def __len__(self):
        return len(self.json_data)
    
    def load_personalized_feature(self, data_file):
        data = np.load(data_file, allow_pickle=True)

        if isinstance(data, np.ndarray) and isinstance(data[0], dict):
            return {entry["id"]: entry["embedding"] for entry in data}
        else:
            raise ValueError("Unexpected data format in the .npy file. Ensure it contains a list of dictionaries.")

    def pad_or_truncate(self, feature, max_len=16):
            """Fill or truncate the input feature sequence"""
            if feature.shape[0] < max_len:
                padding = torch.zeros((max_len - feature.shape[0], feature.shape[1]))
                feature = torch.cat((feature, padding), dim=0)
            else:
                feature = feature[:max_len]
            return feature