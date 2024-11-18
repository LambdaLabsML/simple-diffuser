

# A Very Simple PyTorch Script For Multi-node Training Of Diffusion Model

* Launch your training using `torchrun` -- no need for dependencies such as HF accelerator. 
* Support multi-node


## Prerequisites

* Make sure all nodes can ssh into each other by `hostname`. You can add hostname resolution to `/etc/hosts`.
* Install the following dependencies and clone the repo to the shared storage.

```
export SHARED_STORAGE= # Add the name of your shared storage here. e.g. /home/ubuntu/shared
cd $SHARED_STORAGE && \
git clone https://github.com/LambdaLabsML/simple-diffuser.git && \
cd simple-diffuser && \
pip install -r requirements.txt
```

Note: the above `requirements.txt` works out-of-the-box on Lambda stack. You can also create your own virtual environment and use the `requirements_venv.txt` to install PyTorch related dependencies.

## Usage

### Training
Run the following command from all nodes

```
export SHARED_STORAGE=
export NUM_NODES=
export NUM_GPU_PER_NODE=
export MASTER_NODE_IP=
export MASTER_PORT=

export MODEL_NAME=CompVis/stable-diffusion-v1-4
export DATASET_NAME=lambdalabs/naruto-blip-captions
export HF_HOME=$SHARED_STORAGE/.cache
export TORCHELASTIC_ERROR_FILE=$SHARED_STORAGE/simple-diffuser/error.json
export OMP_NUM_THREADS=1

torchrun \
    --rdzv-id multi-node \
    --rdzv-backend c10d \
    --rdzv-endpoint $MASTER_NODE_IP:$MASTER_PORT \
    --nnodes $NUM_NODES \
    --nproc-per-node $NUM_GPU_PER_NODE \
    --redirects 3 \
    --log-dir $SHARED_STORAGE/simple-diffuser/logs \
    $SHARED_STORAGE/simple-diffuser/train_diffuser.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_name=$DATASET_NAME \
    --resolution=512 --center_crop --random_flip \
    --train_batch_size=16 \
    --max_train_steps=50000 \
    --checkpointing_steps=500 \
    --learning_rate=1e-05 \
    --max_grad_norm=1 \
    --lr_scheduler="constant" --lr_warmup_steps=0 \
    --output_dir="sd-naruto-model" \
    --allow_tf32 \
    --dataloader_num_workers=16 \
    --enable_xformers_memory_efficient_attention
```

### Inference

```
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel

model_path = "./sd-naruto-model/checkpoint-<steps>"
unet = UNet2DConditionModel.from_pretrained(model_path, torch_dtype=torch.float16)

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    unet=unet,
    torch_dtype=torch.float16,
    safety_checker = None,
    requires_safety_checker = False)
pipe.to("cuda")

image = pipe(prompt="Yoda").images[0]
image.save("yoda-ninja.png")
```

You will get something like the image below:

![Example](./yoda-ninja.png)
