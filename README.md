
https://github.com/huggingface/datasets/pull/6883

```
pip install Pillow==9.4.0
```

```
export MODEL_NAME=CompVis/stable-diffusion-v1-4
export DATASET_NAME=lambdalabs/naruto-blip-captions
export HF_HOME=../../.cache
export TORCHELASTIC_ERROR_FILE=./error.json
export OMP_NUM_THREADS=1

rm -rf logs && \
torchrun \
    --rdzv-id multi-node \
    --rdzv-backend c10d \
    --rdzv-endpoint 127.0.0.1:5001 \
    --nnodes 1 \
    --nproc-per-node gpu \
    --redirects 3 \
    --log-dir ./logs \
    train_diffuser.py \
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


```
export MODEL_NAME=CompVis/stable-diffusion-v1-4
export DATASET_NAME=lambdalabs/naruto-blip-captions
export HF_HOME=../../.cache
export TORCHELASTIC_ERROR_FILE=./error.json
export OMP_NUM_THREADS=1

torchrun \
    --standalone \
    --nnodes 1 \
    --nproc-per-node gpu \
    --redirects 3 \
    --log-dir ./logs \
    train_diffuser.py \
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





```
export TORCHELASTIC_ERROR_FILE=./error.json
export OMP_NUM_THREADS=1
torchrun --standalone \
    --nnodes 1 \
    --nproc-per-node gpu \
    --redirects 3 \
    --log-dir ./logs \
    train_llm.py \
    --experiment-name gpt2-alpaca-multi-gpu-$(date +%Y-%m-%dT%H-%M-%S) \
    --dataset-name tatsu-lab/alpaca \
    --model-name openai-community/gpt2
```