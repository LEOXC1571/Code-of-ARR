import torch
from torch.utils.data import Sampler
from verl.experimental.dataset.sampler import AbstractSampler
import random
from collections import defaultdict

class DistSourceIndexSampler(AbstractSampler):
    def __init__(self, data_source, data_config):
        seed = data_config.get("seed", 1)
        torch.manual_seed(seed)
        self.seed = seed
        self.rng = random.Random(seed)
        self.data_source = data_source
        self.num_samples = data_config.val_sampler.get("num_samples", 512)
        print('Using Distributed Source Index Sampling with seed', self.seed)
        print('Data source length:', len(data_source))
        
        source_to_indices = defaultdict(list)
        for idx in range(len(data_source)):
            item = data_source[idx]
            source = item["data_source"]
            source_to_indices[source].append(idx)

        sampled_indices = []
        for source, indices in source_to_indices.items():
            if len(indices) < self.num_samples:
                print('Warning: Not enough samples from {}'.format(source), flush=True)
            k = min(self.num_samples, len(indices))
            print(f'Sampling from source: {source}, total samples: {len(indices)}, extract up to {k}')
            sampled = self.rng.sample(indices, k)
            sampled_indices.extend(sampled)

        self.indices = sampled_indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)