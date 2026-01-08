# Adversarial Yet Cooperative: Multi-Perspective Reasoning in Retrieved-Augmented Language Models

![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.7.1-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Overview

This repository contains the official implementation of **Adversarial Yet Cooperative: Multi-Perspective Reasoning in Retrieved-Augmented Language Models (ARR)**, a novel framework that enhances language models through retrieval-augmented generation.

Our approach combines training with cooperative multi-agent systems, where the Reasoner and the Verifier engage in adversarial yet cooperative dialogue.

## Requirements

- Python 3.10+
- CUDA 12.1+ (for GPU training)
- vllm==0.10.1.1 (support Qwen3 backbone model)

## Installation

### Training Environment

To set up the training environment, run the following commands:

```bash
# Create and activate conda environment
conda create -n arr python=3.10
conda activate arr

# Install PyTorch with CUDA support
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu129

# Install vLLM for fast inference
pip3 install vllm==0.10.1.1

# Install Flash Attention for efficiency
pip3 install flash-attn==2.7.4 --no-build-isolation

# Install Weights & Biases for experiment tracking
pip install wandb
```

### Retriever Environment

For the retrieval system, we recommend using a separate environment:

```bash
# Create and activate conda environment
conda create -n retriever python=3.10
conda activate retriever

# Install PyTorch with CUDA support for FAISS-GPU
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia

# Install transformers and related packages
pip install transformers datasets pyserini

# Install GPU version of FAISS for efficient retrieval
conda install -c pytorch -c nvidia faiss-gpu=1.8.0

# Install FastAPI and Uvicorn for API services
pip install uvicorn fastapi
```

## Usage

### Download the indexing and corpus.
```bash
save_path=/the/path/to/save
python scripts/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
```

### Process the dataset.
```bash
bash scripts/process_merged_dataset.sh
```

### Running Retrieval Systems

Start the retrieval system with:

```bash
# Launch retrieval services
./retrieval_launch.sh
```

This will start the following services:
- Retrieval server on port 8000

You can customize the ports and other settings by modifying the script or passing environment variables.

### Training Models

The repository provides several training scripts for different model sizes. Before running these scripts, ensure you have:

1. Prepared your dataset
2. Configured the environment variables
3. Set up the necessary hardware resources

```bash
# Train Qwen2.5-3B model
./train_qwen2.5-3b.sh

# Train Qwen2.5-7B model
./train_qwen2.5-7b.sh

# Train Qwen3-8B model
./train_qwen3-8b.sh
```



## Project Structure

```
Code-of-ARR/
├── README.md                           # This file
├── train_qwen2.5-3b.sh                 # Training script for Qwen2.5-3B
├── train_qwen2.5-7b.sh                 # Training script for Qwen2.5-7B
├── train_qwen3-8b.sh                   # Training script for Qwen3-8B
├── retrieval_launch.sh                 # Script to launch retrieval services
├── reason_search/                      # Retrieval and search modules
│   ├── llm_agent/                      # LLM agent implementations
│   └── search/                         # Search engine implementations
├── scripts/                            # Utility scripts
│   ├── download.py                     # Data downloading utilities
│   └── data_process/                   # Data processing scripts
└── verl/                               # Main VERL framework
    ├── models/                         # Model implementations
    ├── trainer/                        # Training infrastructure
    ├── workers/                        # Worker implementations
    └── tools/                          # Various tools and utilities
```

