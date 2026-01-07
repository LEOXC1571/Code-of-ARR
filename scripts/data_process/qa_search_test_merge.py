import re
import os
import datasets
from verl.utils.hdfs_io import copy, makedirs
import argparse
from .reasoner_verifier_prefix import make_prefix as make_prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/nq_search')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='base')
    parser.add_argument('--data_dir', type=str, default='nq')
    parser.add_argument('--data_sources', default='nq')

    args = parser.parse_args()
    data_dir = args.data_dir
    data_sources = args.data_sources.split(',')
    all_dataset = []

    for data_source in data_sources:

        if data_source != 'strategyqa':
            dataset = datasets.load_dataset(data_dir, data_source)
        else:
            dataset = datasets.load_dataset('json', data_files="data/strategyqa/test_correct.jsonl")

        if 'test' in dataset:
            print(f'Using the {data_source} test dataset...')
            test_dataset = dataset['test']
        elif 'dev' in dataset:
            print(f'Using the {data_source} dev dataset...')
            test_dataset = dataset['dev']
        else:
            print(f'Using the {data_source} train dataset...')
            test_dataset = dataset['train']

        def make_map_fn(split):
            def process_fn(example, idx):
                example['question'] = example['question'].strip()
                if example['question'][-1] != '?':
                    example['question'] += '?'
                question = make_prefix(example, template_type=args.template_type)
                solution = {
                    "target": example['golden_answers'],
                }

                data = {
                    "data_source": data_source,
                    "prompt": [{
                        "role": "user",
                        "content": question,
                    }],
                    "ability": "fact-reasoning",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": solution
                    },
                    "extra_info": {
                        'split': split,
                        'index': idx,
                    }
                }
                return data
            return process_fn

        test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True, load_from_cache_file=False)
        all_dataset.append(test_dataset)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
    all_test_dataset = datasets.concatenate_datasets(all_dataset)
    
    if args.template_type == 'verifier':
        all_test_dataset.to_parquet(os.path.join(local_dir, 'ver_test.parquet'))
    else:
        all_test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
