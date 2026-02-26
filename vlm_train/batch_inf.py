import argparse
import json
import random
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import ViTImageProcessor

from vlm_train.nn_arch.lm_to_vlm import LM_2_VLM


def parse_args():
    p=argparse.ArgumentParser(
    description="Run batch inference and save predictions with reference captions."
    )
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
    "--checkpoint-dir",
    type=str,
    default="models/from_pod/vlm_peft/best",
    )
    p.add_argument(
    "--qformer-model-path",
    type=str,
    default="models/from_pod/trained_qformer_50k_unimodal_fresh/best",
    )
    p.add_argument(
    "--model-name",
    type=str,
    default="HuggingFaceTB/SmolLM-135M-Instruct",
    )
    p.add_argument(
    "--prompt",
    type=str,
    default="Describe this image in one sentence.",
    )
    p.add_argument("--max-new-tokens",type=int,default=48)
    p.add_argument("--num-samples",type=int,default=50)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument(
    "--out-path",
    type=str,
    default="inference_results/val2017_50_preds.jsonl",
    )
    return p.parse_args()


def generate_one(model,processor,image_path:Path,prompt:str,max_new_tokens:int,device):
    image=Image.open(image_path).convert("RGB")
    pixel_values=processor(images=image,return_tensors="pt")["pixel_values"].to(device)

    with torch.no_grad():
        image_features=model.vit(pixel_values=pixel_values).last_hidden_state
        query_tokens,_=model.qformer.encode_image(image_features)
        query_tokens=query_tokens.to(model.adapter.weight.dtype)
        vision_embeds=model.adapter(query_tokens)

        prompt_text=f"<|user|>\n{prompt}\n<|assistant|>\n"
        prompt_ids=model.tokenizer(
        prompt_text,return_tensors="pt",add_special_tokens=False
        )["input_ids"].to(device)
        text_embeds=model.llm.get_input_embeddings()(prompt_ids)
        vision_embeds=vision_embeds.to(text_embeds.dtype)

        inputs_embeds=torch.cat([vision_embeds,text_embeds],dim=1)
        attention_mask=torch.ones(
        (1,inputs_embeds.size(1)),dtype=torch.long,device=device
        )

        out_ids=model.llm.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=model.tokenizer.pad_token_id,
        eos_token_id=model.tokenizer.eos_token_id,
        )

    return model.tokenizer.decode(out_ids[0],skip_special_tokens=True).strip()


def main():
    args=parse_args()
    device=torch.device("cuda" if torch.cuda.is_available()else "cpu")
    print(f"Device: {device}")

    manifest_path=Path(args.manifest_path)
    images_dir=Path(args.images_dir)
    out_path=Path(args.out_path)
    out_path.parent.mkdir(parents=True,exist_ok=True)

    with manifest_path.open("r",encoding="utf-8")as f:
        records=[json.loads(line)for line in f if line.strip()]

    if not records:
        raise RuntimeError(f"No records found in {manifest_path}")

    rng=random.Random(args.seed)
    sample_n=min(args.num_samples,len(records))
    sampled=rng.sample(records,sample_n)

    model=LM_2_VLM(
    model_name=args.model_name,
    qformer_model_path=args.qformer_model_path,
    pad_token_id=0,
    train_llm=False,
    ).to(device)
    model.load_checkpoint(args.checkpoint_dir,map_location=str(device))
    if device.type=="cpu":
        model=model.float()
    model.eval()

    processor=ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

    written=0
    skipped=0
    with out_path.open("w",encoding="utf-8")as out_f:
        for rec in tqdm(sampled,desc="Batch inference"):
            image_path=images_dir/rec["file_name"]
            if not image_path.exists():
                skipped+=1
                continue

            pred=generate_one(
            model=model,
            processor=processor,
            image_path=image_path,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            device=device,
            )
            row={
            "image_id":rec.get("image_id"),
            "file_name":rec["file_name"],
            "prediction":pred,
            "captions":rec.get("captions",[]),
            }
            out_f.write(json.dumps(row,ensure_ascii=False)+"\n")
            written+=1

    print(f"Saved {written} predictions to {out_path} (skipped={skipped}).")


if __name__=="__main__":
    main()
