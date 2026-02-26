import argparse
import json
import random
import textwrap
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import DistilBertTokenizerFast,ViTImageProcessor,ViTModel

from vlm_train.nn_arch.qformer import QFormer


def parse_args():
    p=argparse.ArgumentParser(description="Compute I2T/T2I retrieval Recall@K.")
    p.add_argument(
    "--manifest-path",
    type=str,
    default="dataset/coco_subsets/val2017_1k.jsonl",
    )
    p.add_argument(
    "--images-dir",
    type=str,
    default="dataset/coco_subsets/val2017_1k_images",
    )
    p.add_argument(
    "--qformer-path",
    type=str,
    default="models/from_pod/trained_qformer_50k_unimodal_fresh/best",
    )
    p.add_argument("--num-samples",type=int,default=500)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--batch-size",type=int,default=16)
    p.add_argument(
    "--out-json",
    type=str,
    default="inference_results/retrieval_val2017_500_metrics.json",
    )
    p.add_argument(
    "--save-grid",
    action="store_true",
    help="Save an 8x8 similarity grid image.",
    )
    p.add_argument(
    "--grid-path",
    type=str,
    default="inference_results/similarity_grid.jpg",
    )
    return p.parse_args()


def load_records(path:Path):
    rows=[]
    with path.open("r",encoding="utf-8")as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            rec=json.loads(line)
            caps=rec.get("captions")or[]
            caps=[c.strip()for c in caps if isinstance(c,str)and c.strip()]
            if not caps:
                continue
            rows.append(
            {
            "image_id":rec.get("image_id"),
            "file_name":rec["file_name"],
            "caption":caps[0], #1 caption per img for 1-1 retrieval
            }
            )
    return rows


def compute_recall(sim:torch.Tensor,k:int):
    n=sim.size(0)
    # image -> text
    topk_i2t=sim.topk(k,dim=1).indices
    gt=torch.arange(n,device=sim.device).unsqueeze(1)
    i2t=(topk_i2t==gt).any(dim=1).float().mean().item()
    # text -> image
    topk_t2i=sim.topk(k,dim=0).indices.t()
    t2i=(topk_t2i==gt).any(dim=1).float().mean().item()
    return i2t,t2i


def maybe_save_grid(
sim:np.ndarray,
path:Path,
rows_subset:list[dict],
images_dir:Path,
metrics:dict,
eval_n:int,
):
    import matplotlib.pyplot as plt
    from matplotlib import patches

    path.parent.mkdir(parents=True,exist_ok=True)
    g=sim.shape[0]

    #layout top thumbnails, left labels integrated in heatmap
    fig=plt.figure(figsize=(13,14),facecolor="#efefef")
    gs=fig.add_gridspec(
    nrows=3,
    ncols=1,
    height_ratios=[1.4,12,1.4],
    hspace=0.08,
    )

    # Top thumbnails
    top_gs=gs[0].subgridspec(1,g,wspace=0.08)
    for i in range(g):
        ax_img=fig.add_subplot(top_gs[0,i])
        img_path=images_dir/rows_subset[i]["file_name"]
        img=Image.open(img_path).convert("RGB")
        ax_img.imshow(img)
        ax_img.set_title(f"Image {i}",fontsize=10,pad=5)
        ax_img.axis("off")

    ax=fig.add_subplot(gs[1])
    im=ax.imshow(sim,cmap="Blues",aspect="auto",vmin=float(sim.min()),vmax=float(sim.max()))
    ax.set_xticks(np.arange(g))
    ax.set_xticklabels([f"Image {i}" for i in range(g)],fontsize=10)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x",pad=8)

    y_labels=[]
    for rec in rows_subset:
        y_labels.append(textwrap.fill(rec["caption"],width=24))
    ax.set_yticks(np.arange(g))
    ax.set_yticklabels(y_labels,fontsize=10)

    for r in range(g):
        for c in range(g):
            v=sim[r,c]
            txt_color="white" if v>(sim.min()+sim.max())/2 else "black"
            ax.text(c,r,f"{v:.3f}",ha="center",va="center",fontsize=11,color=txt_color)
            if r==c:
                rect=patches.Rectangle(
                (c-0.5,r-0.5),
                1,
                1,
                fill=False,
                edgecolor="blue",
                linewidth=2.0,
                )
                ax.add_patch(rect)

    ax.set_xticks(np.arange(-0.5,g,1),minor=True)
    ax.set_yticks(np.arange(-0.5,g,1),minor=True)
    ax.grid(which="minor",color="black",linestyle="-",linewidth=0.6)
    ax.tick_params(which="minor",bottom=False,left=False)

    ax_footer=fig.add_subplot(gs[2])
    ax_footer.axis("off")
    footer=(
    f"Eval on {eval_n} samples:\n"
    f"Img2Text: R@1: {metrics['i2t_r1']:.4f} | R@5: {metrics['i2t_r5']:.4f} | R@10: {metrics['i2t_r10']:.4f}\n"
    f"Text2Img: R@1: {metrics['t2i_r1']:.4f} | R@5: {metrics['t2i_r5']:.4f} | R@10: {metrics['t2i_r10']:.4f}"
    )
    ax_footer.text(0.01,0.5,footer,fontsize=12,va="center",ha="left")

    fig.savefig(path,dpi=180,bbox_inches="tight")
    plt.close(fig)


def main():
    args=parse_args()
    device=torch.device("cuda" if torch.cuda.is_available()else "cpu")
    print(f"Device:{device}")

    manifest_path=Path(args.manifest_path)
    images_dir=Path(args.images_dir)
    out_json=Path(args.out_json)
    out_json.parent.mkdir(parents=True,exist_ok=True)

    rows=load_records(manifest_path)
    if not rows:
        raise RuntimeError(f"No valid rows in {manifest_path}")

    rng=random.Random(args.seed)
    n=min(args.num_samples,len(rows))
    rows=rng.sample(rows,n)
    print(f"Loaded {n} samples for retrieval eval.")

    image_processor=ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    vit=ViTModel.from_pretrained("google/vit-base-patch16-224").to(device).eval()
    tokenizer=DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    qformer=QFormer.from_pretrained(args.qformer_path).to(device).eval()

    image_embs=[]
    text_embs=[]
    used_rows=[]

    num_batches=(n+args.batch_size-1)//args.batch_size
    with torch.no_grad():
        for start in tqdm(range(0,n,args.batch_size),total=num_batches,desc="Embedding batches"):
            batch=rows[start:start+args.batch_size]
            images=[]
            texts=[]
            valid_rows=[]
            for rec in batch:
                img_path=images_dir/rec["file_name"]
                if not img_path.exists():
                    continue
                images.append(Image.open(img_path).convert("RGB"))
                texts.append(rec["caption"])
                valid_rows.append(rec)
            if not images:
                continue

            pixel_values=image_processor(images=images,return_tensors="pt")[
            "pixel_values"
            ].to(device)
            image_features=vit(pixel_values=pixel_values).last_hidden_state

            tok=tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            )
            input_ids=tok["input_ids"].to(device)
            attn=tok["attention_mask"].to(device)

            img_pool,txt_pool=qformer(
            image_features=image_features,
            text_input_ids=input_ids,
            text_attention_mask=attn,
            attention_mode="uni_modal",
            )
            image_embs.append(F.normalize(img_pool,dim=-1).cpu())
            text_embs.append(F.normalize(txt_pool,dim=-1).cpu())
            used_rows.extend(valid_rows)

    if not image_embs or not text_embs:
        raise RuntimeError("No embeddings produced.check image path and inputs")

    print("Building similarity mat. and computing Recall@K...")
    image_mat=torch.cat(image_embs,dim=0)
    text_mat=torch.cat(text_embs,dim=0)
    m=min(image_mat.size(0),text_mat.size(0))
    image_mat=image_mat[:m]
    text_mat=text_mat[:m]

    sim=image_mat@text_mat.t()

    i2t_r1,t2i_r1=compute_recall(sim,1)
    i2t_r5,t2i_r5=compute_recall(sim,5)
    i2t_r10,t2i_r10=compute_recall(sim,10)

    metrics={
    "num_samples_used":int(m),
    "i2t_r1":i2t_r1,
    "i2t_r5":i2t_r5,
    "i2t_r10":i2t_r10,
    "t2i_r1":t2i_r1,
    "t2i_r5":t2i_r5,
    "t2i_r10":t2i_r10,
    }

    if args.save_grid:
        g=min(8,m)
        sim_grid=sim[:g,:g].t().numpy()
        maybe_save_grid(
        sim=sim_grid,
        path=Path(args.grid_path),
        rows_subset=used_rows[:g],
        images_dir=images_dir,
        metrics=metrics,
        eval_n=m,
        )
        metrics["similarity_grid"]=args.grid_path

    with out_json.open("w",encoding="utf-8")as f:
        json.dump(metrics,f,indent=2)

    print(json.dumps(metrics,indent=2))
    print(f"Saved retrieval metrics to: {out_json}")


if __name__=="__main__":
    main()
