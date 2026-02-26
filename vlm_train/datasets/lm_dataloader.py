import json
import random
from pathlib import Path
from typing import Optional

from PIL import Image
import torch
from torch.utils.data import DataLoader,Dataset,random_split
from transformers import ViTImageProcessor


class LmCocoCaptionDataset(Dataset):
    def __init__(
    self,
    manifest_path:Path,
    images_dir:Path,
    prefix_template:str="Describe this image in one sentence.",
    ):
        self.manifest_path=Path(manifest_path)
        self.images_dir=Path(images_dir)
        self.prefix_template=prefix_template

        with self.manifest_path.open("r",encoding="utf-8")as f:
            self.records=[json.loads(line)for line in f]

    def __len__(self):
        return len(self.records)

    def __getitem__(self,idx):
        rec=self.records[idx]
        image_path=self.images_dir/rec["file_name"]
        image=Image.open(image_path).convert("RGB")
        caption=random.choice(rec["captions"]).strip()
        return{
        "image":image,
        "prefix":self.prefix_template,
        "assistant_prompt":caption,
        }


def get_dataloader(
batch_size:int,
tokenizer_name:str,
manifest_path:str="dataset/coco_subsets/train2017_50k.jsonl",
images_dir:str="dataset/coco_subsets/train2017_50k_images",
image_processor_name:str="google/vit-base-patch16-224",
val_ratio:float=0.1,
num_workers:int=0,
seed:int=42,
subset_size:Optional[int]=None,
):
# tokenizer_name kept in signature for compatibility with the referenced code path.
    _=tokenizer_name

    dataset=LmCocoCaptionDataset(
    manifest_path=Path(manifest_path),
    images_dir=Path(images_dir),
    )
    if subset_size is not None and subset_size<len(dataset):
        indices=list(range(subset_size))
        from torch.utils.data import Subset

        dataset=Subset(dataset,indices)

    val_size=max(1,int(len(dataset)*val_ratio))
    train_size=len(dataset)-val_size
    train_ds,val_ds=random_split(
    dataset,
    [train_size,val_size],
    generator=torch.Generator().manual_seed(seed),
    )

    processor=ViTImageProcessor.from_pretrained(image_processor_name)

    def collate_fn(batch):
        images=[b["image"]for b in batch]
        pixel_values=processor(images=images,return_tensors="pt")["pixel_values"]
        return{
        "pixel_values":pixel_values,
        "prefix":[b["prefix"]for b in batch],
        "assistant_prompt":[b["assistant_prompt"]for b in batch],
        }

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
    return train_loader,val_loader
