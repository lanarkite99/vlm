import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader,random_split
from tqdm import tqdm
from transformers import(
DistilBertModel,
DistilBertTokenizerFast,
ViTImageProcessor,
ViTModel,
)

from vlm_train.datasets.coco_subset_dataset import CocoSubsetCaptionDataset
from vlm_train.nn_arch.qformer import QFormer


def set_seed(seed:int=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calc_clip_loss(image_emb,text_emb,tau:float=0.07):
    n=image_emb.size(0)
    image_emb=F.normalize(image_emb,dim=-1)
    text_emb=F.normalize(text_emb,dim=-1)
    logits=image_emb@text_emb.t()/tau
    labels=torch.arange(n,device=logits.device)
    loss_i2t=F.cross_entropy(logits,labels)
    loss_t2i=F.cross_entropy(logits.t(),labels)
    return 0.5*(loss_i2t+loss_t2i)


def run_eval(qformer,vit,loader,device,max_batches=20):
    qformer.eval()
    losses=[]
    with torch.no_grad():
        for i,batch in enumerate(loader):
            pixel_values=batch["pixel_values"].to(device)
            input_ids=batch["input_ids"].to(device)
            attention_mask=batch["attention_mask"].to(device)

            image_features=vit(pixel_values=pixel_values).last_hidden_state
            image_proj,text_proj=qformer(
            image_features=image_features,
            text_input_ids=input_ids,
            text_attention_mask=attention_mask,
            attention_mode="multi_modal",
            )
            losses.append(calc_clip_loss(image_proj,text_proj).item())
            if(i+1)>=max_batches:
                break
    return float(np.mean(losses))if losses else float("nan")


def build_collate_fn(vit_processor,tokenizer,max_text_len:int=32):
    def collate_fn(batch):
        images=[sample["image"]for sample in batch]
        captions=[sample["caption"]for sample in batch]

        vit_inputs=vit_processor(images=images,return_tensors="pt")
        text_inputs=tokenizer(
        captions,
        padding=True,
        truncation=True,
        max_length=max_text_len,
        return_tensors="pt",
        )

        return{
        "pixel_values":vit_inputs["pixel_values"],
        "input_ids":text_inputs["input_ids"],
        "attention_mask":text_inputs["attention_mask"],
        }

    return collate_fn


def save_training_state(
state_path:Path,
qformer:QFormer,
optimizer:optim.Optimizer,
epoch:int,
global_step:int,
best_val_loss:float,
):
    state_path.parent.mkdir(parents=True,exist_ok=True)
    state={
    "epoch":epoch,
    "global_step":global_step,
    "best_val_loss":best_val_loss,
    "qformer_state_dict":qformer.state_dict(),
    "optimizer_state_dict":optimizer.state_dict(),
    "python_rng_state":random.getstate(),
    "numpy_rng_state":np.random.get_state(),
    "torch_rng_state":torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_rng_state_all"]=torch.cuda.get_rng_state_all()
    torch.save(state,state_path)


def load_training_state(
state_path:Path,
qformer:QFormer,
optimizer:optim.Optimizer,
device:torch.device,
):
    state=torch.load(state_path,map_location=device)
    qformer.load_state_dict(state["qformer_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])

    if "python_rng_state" in state:
        random.setstate(state["python_rng_state"])
    if "numpy_rng_state" in state:
        np.random.set_state(state["numpy_rng_state"])
    if "torch_rng_state" in state:
        torch.set_rng_state(state["torch_rng_state"])
    if torch.cuda.is_available()and "torch_cuda_rng_state_all" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda_rng_state_all"])

    start_epoch=int(state["epoch"])+1
    global_step=int(state["global_step"])
    best_val_loss=float(state["best_val_loss"])
    return start_epoch,global_step,best_val_loss


def parse_args():
    parser=argparse.ArgumentParser(description="Train Q-Former with contrastive objective")
    parser.add_argument(
    "--manifest-path",
    type=Path,
    default=Path("dataset/coco_subsets/train2017_256.jsonl"),
    )
    parser.add_argument(
    "--images-dir",
    type=Path,
    default=Path("dataset/coco_subsets/train2017_256_images"),
    )
    parser.add_argument("--model-id",type=str,default="trained_qformer")
    parser.add_argument("--epochs",type=int,default=3)
    parser.add_argument("--batch-size",type=int,default=8)
    parser.add_argument("--lr",type=float,default=1e-4)
    parser.add_argument("--max-text-len",type=int,default=32)
    parser.add_argument("--num-workers",type=int,default=0)
    parser.add_argument("--resume",action="store_true",help="Resume from latest training state")
    parser.add_argument(
    "--state-path",
    type=Path,
    default=None,
    help="Optional explicit path to training state checkpoint",
    )
    return parser.parse_args()


def main():
    args=parse_args()
    set_seed(42)
    device=torch.device("cuda" if torch.cuda.is_available()else "cpu")
    print(f"Device: {device}")

    manifest_path=args.manifest_path
    images_dir=args.images_dir
    model_id=args.model_id
    epochs=args.epochs
    batch_size=args.batch_size
    lr=args.lr
    max_text_len=args.max_text_len
    num_workers=args.num_workers

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")

    vit_name="google/vit-base-patch16-224"
    bert_name="distilbert-base-uncased"

    vit_processor=ViTImageProcessor.from_pretrained(vit_name)
    tokenizer=DistilBertTokenizerFast.from_pretrained(bert_name)
    vit=ViTModel.from_pretrained(vit_name).to(device)
    vit.eval()
    for p in vit.parameters():
        p.requires_grad=False

    bert=DistilBertModel.from_pretrained(bert_name)
    qformer=QFormer(bert_model=bert,n_queries=32,cross_every=2,n_heads=12).to(device)

    full_ds=CocoSubsetCaptionDataset(
    manifest_path=manifest_path,
    images_dir=images_dir,
    transform=None,
    )
    val_size=max(1,int(0.1*len(full_ds)))
    train_size=len(full_ds)-val_size
    train_ds,val_ds=random_split(
    full_ds,
    [train_size,val_size],
    generator=torch.Generator().manual_seed(42),
    )

    collate_fn=build_collate_fn(vit_processor,tokenizer,max_text_len=max_text_len)
    train_loader=DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    collate_fn=collate_fn,
    )
    val_loader=DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    collate_fn=collate_fn,
    )

    grouped_params=qformer.get_grouped_parameters()
    optimizer=optim.AdamW(
    [
    {"params":grouped_params["default"],"lr":lr*0.1},
    {"params":grouped_params["cross_blocks"],"lr":lr},
    {"params":grouped_params["query_embeddings"],"lr":lr},
    ],
    weight_decay=1e-4,
    )

    save_root=Path("models")/model_id
    os.makedirs(save_root,exist_ok=True)
    state_path=args.state_path if args.state_path is not None else(save_root/"latest_training_state.pt")

    start_epoch=0
    best_val_loss=float("inf")
    global_step=0
    if args.resume:
        if not state_path.exists():
            raise FileNotFoundError(f"Resume requested but state checkpoint not found: {state_path}")
        start_epoch,global_step,best_val_loss=load_training_state(
        state_path=state_path,
        qformer=qformer,
        optimizer=optimizer,
        device=device,
        )
        print(
        f"Resumed from {state_path} at epoch={start_epoch + 1} "
        f"global_step={global_step} best_val_loss={best_val_loss:.4f}"
        )

    for epoch in range(start_epoch,epochs):
        qformer.train()
        running_losses=[]
        pbar=tqdm(train_loader,desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            global_step+=1
            pixel_values=batch["pixel_values"].to(device)
            input_ids=batch["input_ids"].to(device)
            attention_mask=batch["attention_mask"].to(device)

            with torch.no_grad():
                image_features=vit(pixel_values=pixel_values).last_hidden_state

            image_proj,text_proj=qformer(
            image_features=image_features,
            text_input_ids=input_ids,
            text_attention_mask=attention_mask,
            attention_mode="multi_modal",
            )
            loss=calc_clip_loss(image_proj,text_proj)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss=float(np.mean(running_losses))
        val_loss=run_eval(qformer,vit,val_loader,device=device,max_batches=20)
        print(
        f"Epoch {epoch + 1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} step={global_step}"
        )

        qformer.save_pretrained(str(save_root/"latest"))
        if val_loss<best_val_loss:
            best_val_loss=val_loss
            qformer.save_pretrained(str(save_root/"best"))
            print(f"Saved best checkpoint with val_loss={val_loss:.4f}")
        save_training_state(
        state_path=state_path,
        qformer=qformer,
        optimizer=optimizer,
        epoch=epoch,
        global_step=global_step,
        best_val_loss=best_val_loss,
        )


if __name__=="__main__":
    main()
