import torch
from torch.utils.data import Sampler
from verl.experimental.dataset.sampler import AbstractSampler

class GlobalIndexSampler(AbstractSampler):
    def __init__(self, data_source, data_config):
        """
        indices: 全局随机索引列表 (e.g., [5, 2, 9, ...])
        """
        seed = data_config.get("seed", 1)
        torch.manual_seed(seed)
        self.indices = torch.randperm(len(data_source)).tolist()

    def __iter__(self):
        return iter(self.indices)
    def __len__(self):
        return len(self.indices)