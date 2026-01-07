# Adversarial Yet Cooperative: Multi-Perspective Reasoning in Retrieved-Augmented Language Models


## Installation

### Training environment
```bash
conda create -n arr python=3.10
conda activate arr
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu129
# install vllm
pip3 install vllm==0.10.1.1

pip3 install flash-attn==2.7.4 --no-build-isolation

pip install wandb
```

### Retriever environment
We recommend using a seperate environment.
```bash
conda create -n retriever python=3.10
conda activate retriever

# we recommend installing torch with conda for faiss-gpu
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets pyserini

## install the gpu version faiss to guarantee efficient RL rollout
conda install -c pytorch -c nvidia faiss-gpu=1.8.0

## API function
pip install uvicorn fastapi
```
