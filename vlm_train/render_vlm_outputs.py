import argparse
import json
import random
import textwrap
from pathlib import Path

from PIL import Image,ImageDraw,ImageFont


def fit_wrapped_text(draw,text,font_path,max_w,max_h,start_size=44,min_size=16):
    for size in range(start_size,min_size-1,-2):
        try:
            font=ImageFont.truetype(font_path,size)
        except Exception:
            font=ImageFont.load_default()
        # conservative wrap width by character count
        for wrap in range(28,10,-1):
            wrapped=textwrap.fill(text,width=wrap)
            bbox=draw.multiline_textbbox((0,0),wrapped,font=font,spacing=6)
            w=bbox[2]-bbox[0]
            h=bbox[3]-bbox[1]
            if w<=max_w and h<=max_h:
                return wrapped,font
    return textwrap.fill(text,width=12),ImageFont.load_default()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--preds-jsonl",type=str,default="inference_results/val2017_500_preds.jsonl")
    p.add_argument("--images-dir",type=str,default="dataset/coco_subsets/val2017_1k_images")
    p.add_argument("--out",type=str,default="inference_results/vlm_outputs.jpg")
    p.add_argument("--num-samples",type=int,default=6)
    p.add_argument("--seed",type=int,default=42)
    args=p.parse_args()

    preds=Path(args.preds_jsonl)
    images_dir=Path(args.images_dir)
    out=Path(args.out)
    out.parent.mkdir(parents=True,exist_ok=True)

    rows=[]
    with preds.open("r",encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r=json.loads(line)
            img_path=images_dir/r["file_name"]
            if img_path.exists():
                rows.append((img_path,r.get("prediction","").strip()))

    random.Random(args.seed).shuffle(rows)
    rows=rows[:args.num_samples]

    W,H=1900,980
    canvas=Image.new("RGB",(W,H),(8,10,14))
    d=ImageDraw.Draw(canvas)

    try:
        f_title=ImageFont.truetype("arial.ttf",28)
        f_label=ImageFont.truetype("arial.ttf",18)
    except Exception:
        f_title=ImageFont.load_default()
        f_label=ImageFont.load_default()

    d.text((20,16),"VLM Outputs (Your Trained Model)",fill=(220,235,255),font=f_title)

    gap=18
    cols=3
    rows_n=2
    cell_w=(W-40-gap*(cols-1))//cols
    cell_h=(H-70-gap*(rows_n-1))//rows_n

    for idx,(img_path,pred) in enumerate(rows):
        r=idx//cols
        c=idx%cols
        x=20+c*(cell_w+gap)
        y=60+r*(cell_h+gap)
        d.rounded_rectangle([x,y,x+cell_w,y+cell_h],radius=14,fill=(12,16,22),outline=(28,39,52),width=2)

        img=Image.open(img_path).convert("RGB")
        iw,ih=img.size
        target_w=int(cell_w*0.47)
        target_h=cell_h-40
        scale=min(target_w/iw,target_h/ih)
        niw,nih=int(iw*scale),int(ih*scale)
        img=img.resize((niw,nih))
        ix=x+20+(target_w-niw)//2
        iy=y+20+(target_h-nih)//2
        canvas.paste(img,(ix,iy))

        tx=x+int(cell_w*0.5)
        ty=y+20
        tw=cell_w-int(cell_w*0.5)-20
        th=cell_h-40
        d.rounded_rectangle([tx,ty,tx+tw,ty+th],radius=10,fill=(6,10,14),outline=(0,180,255),width=2)
        d.text((tx+14,ty+12),"Generated Response",fill=(120,200,255),font=f_label)

        text_y=ty+48
        max_text_w=tw-28
        max_text_h=th-58
        wrapped,font=fit_wrapped_text(d,pred,"consola.ttf",max_text_w,max_text_h,start_size=42,min_size=16)
        d.multiline_text((tx+14,text_y),wrapped,fill=(180,255,180),font=font,spacing=6)

    canvas.save(out,quality=95)
    print(f"Saved: {out}")


if __name__=="__main__":
    main()
