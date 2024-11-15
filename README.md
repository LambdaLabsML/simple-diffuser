

# A Very Simple PyTorch Script For Multi-node Training Of Diffusion Model

* Launch your training using `torchrun` -- no need for dependencies such as HF accelerator. 
* Support multi-node


## Prerequisites

* Make sure all nodes can ssh into each other by `hostname`. You can add hostname resolution to `/etc/hosts`.
* Install the following dependencies and clone the repo to the shared storage.

```
pip install diffusers

# https://github.com/huggingface/datasets/pull/6883
pip install Pillow==9.4.0

export SHARED_STORAGE=
cd $SHARED_STORAGE && \
git clone https://github.com/chuanli11/simple-diffuser.git && \
cd simple-diffuser
```


## Usage
Run the following command from all nodes

```
export NUM_NODES=
export NUM_GPU_PER_NODE=
export MASTER_NODE_IP=
export MASTER_PORT=

export MODEL_NAME=CompVis/stable-diffusion-v1-4
export DATASET_NAME=lambdalabs/naruto-blip-captions
export HF_HOME=/home/ubuntu/$SHARED_STORAGE/.cache
export TORCHELASTIC_ERROR_FILE=/home/ubuntu/$SHARED_STORAGE/simple-diffuser/error.json
export OMP_NUM_THREADS=1

torchrun \
    --rdzv-id multi-node \
    --rdzv-backend c10d \
    --rdzv-endpoint $MASTER_NODE_IP:$MASTER_PORT \
    --nnodes $NUM_NODES \
    --nproc-per-node $NUM_GPU_PER_NODE \
    --redirects 3 \
    --log-dir /home/ubuntu/$SHARED_STORAGE/simple-diffuser/logs \
    /home/ubuntu/$SHARED_STORAGE/simple-diffuser/train_diffuser.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_name=$DATASET_NAME \
    --resolution=512 --center_crop --random_flip \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --gradient_checkpointing \
    --max_train_steps=100 \
    --learning_rate=1e-05 \
    --max_grad_norm=1 \
    --lr_scheduler="constant" --lr_warmup_steps=0 \
    --output_dir="sd-naruto-model"
```