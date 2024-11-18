#!/usr/bin/env python
# coding=utf-8

import argparse
from contextlib import contextmanager
import logging
import math
import os
import random
import shutil
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.checkpoint
import numpy as np
from contextlib import nullcontext
from pathlib import Path
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer
from datasets import load_dataset

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available

import tqdm
import time

logger = logging.getLogger(__name__)


DATASET_NAME_MAPPING = {
    "lambdalabs/naruto-blip-captions": ("image", "text"),
}

def parse_args():
    parser = argparse.ArgumentParser(description="Training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=2000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Run validation every X epochs.",
    )

    args = parser.parse_args()

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ["LOCAL_RANK"])
    else:
        args.rank = 0
        args.world_size = 1
    return args


def cleanup_distributed():
    dist.destroy_process_group()

@contextmanager
def rank0_first():
    rank = dist.get_rank()
    if rank == 0:
        yield
    dist.barrier()
    if rank > 0:
        yield
    dist.barrier()


class LocalTimer:
    def __init__(self, device: torch.device):
        if device.type == "cpu":
            self.synchronize = lambda: torch.cpu.synchronize(device=device)
        elif device.type == "cuda":
            self.synchronize = lambda: torch.cuda.synchronize(device=device)
        self.measurements = []
        self.start_time = None

    def __enter__(self):
        self.synchronize()
        self.start_time = time.time()
        return self

    def __exit__(self, type, value, traceback):
        if traceback is None:
            self.synchronize()
            end_time = time.time()
            self.measurements.append(end_time - self.start_time)
        self.start_time = None

    def avg_elapsed_ms(self):
        return 1000 * (sum(self.measurements) / len(self.measurements))

    def reset(self):
        self.measurements = []
        self.start_time = None


def main():
    args = parse_args()
    
    torch.manual_seed(args.seed)

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    dist.init_process_group()

    rank = dist.get_rank()
    local_rank = rank % torch.cuda.device_count()
    world_size = dist.get_world_size()


    logging.basicConfig(
        format=f"[rank={rank}] [%(asctime)s] %(levelname)s:%(message)s",
        level=logging.INFO,
    )
    logger.info(os.environ)
    logger.info(args)
    logger.info(f"rank={rank} world size={world_size}")

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    with rank0_first():
        with device:
            # Load scheduler, tokenizer and models.
            noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
            tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
            text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
            vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
            unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    unet = DDP(
        unet, device_ids=[local_rank], output_device=local_rank
    )

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train()

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon
    )
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer, args.lr_warmup_steps, args.max_train_steps)

    # Load dataset and create dataloader
    with rank0_first():
      dataset = load_dataset(args.dataset_name, cache_dir=args.cache_dir)
    column_names = dataset["train"].column_names

    dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)

    if args.image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.caption_column is None:
        caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        return examples
    

    train_dataset = dataset["train"].with_transform(preprocess_train)


    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    state = {
        "epoch": 0,
        "global_step": 0,
        "epoch_step": 0,
        "running_loss": 0,
    }

    timers = {k: LocalTimer(device) for k in ["data", "forward", "backward", "update"]}

    # Train loop
    global_step = 0
    for state["epoch"] in range(args.num_train_epochs):
        logger.info(f"Begin epoch {state['epoch']} at step {state['epoch_step']}")
        progress_bar = tqdm.tqdm(range(len(train_dataloader)), disable=rank > 0)
        if state["epoch_step"] > 0:
            progress_bar.update(state["epoch_step"])
        
        batches = iter(train_dataloader)

        for i_step in range(len(train_dataloader)):
            with timers["data"], torch.no_grad():
                batch = next(batches)
                pixel_values, input_ids = batch["pixel_values"].to(device), batch["input_ids"].to(device)
            
            with timers["forward"]:
                latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],)).to(device)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                encoder_hidden_states = text_encoder(input_ids)[0]
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states)[0]
            
            with timers["backward"]:
                loss = F.mse_loss(model_pred.float(), noise.float())
                optimizer.zero_grad()
                loss.backward()

            with timers["update"]:    
                if args.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()

            state["global_step"] += 1
            state["epoch_step"] += 1
            state["running_loss"] += loss.item()
            progress_bar.update(1)

            if args.rank == 0 and state["global_step"] % args.logging_steps == 0:
                samples_per_step = world_size * args.train_batch_size
                ms_per_step = sum(t.avg_elapsed_ms() for t in timers.values())

                info = {
                    "global_step": state["global_step"],
                    "lr": lr_scheduler.get_last_lr()[0],
                    "running_loss": state["running_loss"] / args.logging_steps,
                    "epoch": state["epoch"],
                    "epoch_progress": state["epoch_step"] / len(train_dataloader),
                    "num_batches_remaining": len(train_dataloader) - i_step,
                    "samples/s": 1000 * samples_per_step / ms_per_step,
                    "time/total": ms_per_step,
                    **{
                        f"time/{k}": timer.avg_elapsed_ms()
                        for k, timer in timers.items()
                    },
                }

                logger.info(info)
                state["running_loss"] = 0
                for t in timers.values():
                    t.reset()
                
            # Save checkpoint
            if (args.rank == 0 and state["global_step"] % args.checkpointing_steps == 0) or state["global_step"] >= args.max_train_steps:
                save_path = os.path.join(args.output_dir, "checkpoint-" + str(state["global_step"]))
                os.makedirs(save_path, exist_ok=True)
                unet.module.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)

                if state["global_step"] >= args.max_train_steps:
                    break

    # Final cleanup
    cleanup_distributed()

if __name__ == "__main__":
    main()
