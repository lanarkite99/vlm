import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from accelerate import Accelerator
from tqdm import tqdm
from transformers import AutoTokenizer,get_cosine_schedule_with_warmup

from vlm_train.datasets.lm_dataloader import get_dataloader
from vlm_train.nn_arch.lm_to_vlm import LM_2_VLM


def parse_args():
    parser=argparse.ArgumentParser(description="Stage 2: train VLM with LoRA")
    parser.add_argument(
    "--model-name",
    type=str,
    default="HuggingFaceTB/SmolLM-135M-Instruct",
    )
    parser.add_argument(
    "--qformer-model-path",
    type=str,
    default="models/trained_qformer_50k/best",
    )
    parser.add_argument(
    "--manifest-path",
    type=str,
    default="dataset/coco_subsets/train2017_50k.jsonl",
    )
    parser.add_argument(
    "--images-dir",
    type=str,
    default="dataset/coco_subsets/train2017_50k_images",
    )
    parser.add_argument("--model-id",type=str,default="vlm_peft")
    parser.add_argument("--epochs",type=int,default=5)
    parser.add_argument("--batch-size",type=int,default=8)
    parser.add_argument("--num-workers",type=int,default=0)
    parser.add_argument("--max-text-len",type=int,default=96)
    parser.add_argument("--lr-slow",type=float,default=1e-4)
    parser.add_argument("--lr-fast",type=float,default=5e-4)
    parser.add_argument("--weight-decay",type=float,default=1e-4)
    parser.add_argument("--warmup-steps",type=int,default=100)
    parser.add_argument("--grad-accum-steps",type=int,default=4)
    parser.add_argument("--max-grad-norm",type=float,default=1.0)
    parser.add_argument("--mixed-precision",type=str,default="bf16")
    parser.add_argument("--log-every",type=int,default=20)
    parser.add_argument("--save-every",type=int,default=100)
    parser.add_argument("--eval-batches",type=int,default=20)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--freeze-llm",action="store_true")
    return parser.parse_args()


def run_eval(model,loader,accelerator,max_batches=20):
    model.eval()
    losses=[]
    with torch.no_grad():
        for i,batch in enumerate(loader):
            pixel_values=batch["pixel_values"]
            prefix=batch["prefix"]
            assistant=batch["assistant_prompt"]
            with accelerator.autocast():
                out=model(pixel_values,prefix,assistant)
            loss=accelerator.gather(out.loss.detach()).mean().item()
            losses.append(loss)
            if i+1>=max_batches:
                break
    model.train()
    if not losses:
        return float("inf")
    return float(np.mean(losses))


def save_training_state(path,epoch,step,best_val,optimizer,scheduler):
    torch.save(
    {
    "epoch":epoch,
    "step":step,
    "best_val":best_val,
    "optimizer":optimizer.state_dict(),
    "scheduler":scheduler.state_dict(),
    "torch_rng_state":torch.get_rng_state(),
    "cuda_rng_state":(
    torch.cuda.get_rng_state_all()if torch.cuda.is_available()else None
    ),
    },
    path,
    )


def maybe_load_training_state(path,optimizer,scheduler):
    if not path.exists():
        return 0,0,float("inf")

    state=torch.load(path,map_location="cpu",weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if "torch_rng_state" in state and isinstance(state["torch_rng_state"],torch.Tensor):
        torch.set_rng_state(state["torch_rng_state"])
    if torch.cuda.is_available()and state.get("cuda_rng_state")is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng_state"])
    return int(state["epoch"]),int(state["step"]),float(state["best_val"])


def main():
    args=parse_args()
    accelerator=Accelerator(
    gradient_accumulation_steps=args.grad_accum_steps,
    mixed_precision=args.mixed_precision,
    )
    accelerator.print(f"Device: {accelerator.device}")

    train_loader,val_loader=get_dataloader(
    batch_size=args.batch_size,
    tokenizer_name=args.model_name,
    manifest_path=args.manifest_path,
    images_dir=args.images_dir,
    num_workers=args.num_workers,
    )

    tok=AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token is None:
        tok.pad_token=tok.eos_token

    model=LM_2_VLM(
    model_name=args.model_name,
    qformer_model_path=args.qformer_model_path,
    pad_token_id=tok.pad_token_id,
    max_text_len=args.max_text_len,
    train_llm=not args.freeze_llm,
    )

    grouped=model.get_grouped_params()
    optim_groups=[
    {"params":grouped["qformer_default"],"lr":args.lr_slow},
    {"params":grouped["qformer_cross"],"lr":args.lr_slow},
    {"params":grouped["qformer_query"],"lr":args.lr_slow},
    {"params":grouped["adapter"],"lr":args.lr_fast},
    ]
    if grouped["llm_trainable"]:
        optim_groups.append({"params":grouped["llm_trainable"],"lr":args.lr_fast})

    optimizer=optim.AdamW(optim_groups,weight_decay=args.weight_decay)

    total_steps=(
    len(train_loader)*args.epochs//max(1,args.grad_accum_steps)
    )
    scheduler=get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=args.warmup_steps,
    num_training_steps=total_steps,
    )

    model,optimizer,train_loader,val_loader,scheduler=accelerator.prepare(
    model,optimizer,train_loader,val_loader,scheduler
    )

    save_root=Path("models")/args.model_id
    os.makedirs(save_root,exist_ok=True)
    state_path=save_root/"latest_training_state.pt"

    start_epoch=0
    step=0
    best_val=float("inf")

    if args.resume:
        unwrapped=accelerator.unwrap_model(model)
        unwrapped.load_checkpoint(str(save_root/"latest"),map_location="cpu")
        start_epoch,step,best_val=maybe_load_training_state(
        state_path,optimizer,scheduler
        )
        accelerator.print(
        f"Resumed: epoch={start_epoch}, step={step}, best_val={best_val:.4f}"
        )

    accelerator.print(f"Total train steps: {total_steps}")
    accelerator.print(f"Grad accumulation: {args.grad_accum_steps}")

    for epoch in range(start_epoch,args.epochs):
        pbar=tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{args.epochs}",
        disable=not accelerator.is_local_main_process,
        )

        for batch in pbar:
            with accelerator.accumulate(model):
                pixel_values=batch["pixel_values"]
                prefix=batch["prefix"]
                assistant=batch["assistant_prompt"]

                with accelerator.autocast():
                    out=model(pixel_values,prefix,assistant)
                    loss=out.loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(),args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.is_local_main_process:
                pbar.set_postfix(
                loss=f"{loss.item():.4f}",lr=f"{scheduler.get_last_lr()[0]:.2e}"
                )

            if accelerator.sync_gradients:
                step+=1

                if step%args.log_every==0:
                    val_loss=run_eval(
                    model,val_loader,accelerator,max_batches=args.eval_batches
                    )
                    accelerator.print(
                    f"step {step}: train_loss={loss.item():.4f} val_loss={val_loss:.4f}"
                    )
                    if val_loss<best_val:
                        best_val=val_loss
                        unwrapped=accelerator.unwrap_model(model)
                        unwrapped.save_checkpoint(str(save_root/"best"))
                        accelerator.print(
                        f"Saved best checkpoint with val_loss={best_val:.4f}"
                        )

                if step%args.save_every==0 and accelerator.is_main_process:
                    unwrapped=accelerator.unwrap_model(model)
                    unwrapped.save_checkpoint(str(save_root/"latest"))
                    save_training_state(
                    state_path,
                    epoch=epoch,
                    step=step,
                    best_val=best_val,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    )

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped=accelerator.unwrap_model(model)
            unwrapped.save_checkpoint(str(save_root/"latest"))
            save_training_state(
            state_path,
            epoch=epoch+1,
            step=step,
            best_val=best_val,
            optimizer=optimizer,
            scheduler=scheduler,
            )
        epoch_val=run_eval(model,val_loader,accelerator,max_batches=args.eval_batches)
        accelerator.print(f"Epoch {epoch + 1} done. val_loss={epoch_val:.4f}")
        if epoch_val<best_val and accelerator.is_main_process:
            best_val=epoch_val
            unwrapped=accelerator.unwrap_model(model)
            unwrapped.save_checkpoint(str(save_root/"best"))
            accelerator.print(f"Saved best checkpoint with val_loss={best_val:.4f}")


if __name__=="__main__":
    main()
