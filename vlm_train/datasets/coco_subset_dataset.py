import json
import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class CocoSubsetCaptionDataset(Dataset):
    """
    Expects a JSONL manifest where each line is:
    {
      "image_id": int,
      "file_name": str,
      "captions": [str, ...]
    }
    """

    def __init__(self,manifest_path,images_dir,transform=None):
        self.manifest_path=Path(manifest_path)
        self.images_dir=Path(images_dir)
        self.transform=transform

        with self.manifest_path.open("r",encoding="utf-8")as f:
            self.records=[json.loads(line)for line in f]

    def __len__(self):
        return len(self.records)

    def __getitem__(self,idx):
        rec=self.records[idx]
        image_path=self.images_dir/rec["file_name"]
        image=Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image=self.transform(image)

            # Random caption per sample call: naturally varies across epochs.
        caption=random.choice(rec["captions"])

        return{
        "image_id":rec["image_id"],
        "image":image,
        "caption":caption,
        "file_name":rec["file_name"],
        }
