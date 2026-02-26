import argparse
import json
from pathlib import Path
from statistics import mean

import evaluate


def parse_args():
    p=argparse.ArgumentParser(
    description="Eval caption predictions stored in JSONL"
    )
    p.add_argument(
    "--preds-jsonl",
    type=str,
    default="inference_results/val2017_50_preds.jsonl",
    help="Path to JSONL with fields: prediction, captions",
    )
    p.add_argument(
    "--out-json",
    type=str,
    default="inference_results/val2017_50_metrics.json",
    help="Path to save agg. metrics",
    )
    p.add_argument(
    "--out-csv",
    type=str,
    default="inference_results/val2017_50_metrics_per_sample.csv",
    help="Path to save per sample preds and first reference",
    )
    p.add_argument(
    "--bertscore-model",
    type=str,
    default="microsoft/deberta-xlarge-mnli",
    help="BERTScore model type",
    )
    return p.parse_args()


def load_rows(path:Path):
    rows=[]
    with path.open("r",encoding="utf-8")as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            obj=json.loads(line)
            pred=(obj.get("prediction")or "").strip()
            caps=obj.get("captions")or[]
            caps=[c.strip()for c in caps if isinstance(c,str)and c.strip()]
            if not pred or not caps:
                continue
            rows.append(
            {
            "image_id":obj.get("image_id"),
            "file_name":obj.get("file_name"),
            "prediction":pred,
            "captions":caps,
            }
            )
    return rows


def main():
    args=parse_args()
    preds_path=Path(args.preds_jsonl)
    out_json=Path(args.out_json)
    out_csv=Path(args.out_csv)
    out_json.parent.mkdir(parents=True,exist_ok=True)
    out_csv.parent.mkdir(parents=True,exist_ok=True)

    rows=load_rows(preds_path)
    if not rows:
        raise RuntimeError(f"no valid rows found in {preds_path}")

    predictions=[r["prediction"]for r in rows]
    references_multi=[r["captions"]for r in rows]
    references_single=[r["captions"][0]for r in rows]

    sacrebleu=evaluate.load("sacrebleu")
    rouge=evaluate.load("rouge")
    bertscore=evaluate.load("bertscore")

    bleu_res=sacrebleu.compute(
    predictions=predictions,
    references=references_multi,
    )
    rouge_res=rouge.compute(
    predictions=predictions,
    references=references_single,
    use_stemmer=True,
    )
    bert_res=bertscore.compute(
    predictions=predictions,
    references=references_single,
    lang="en",
    model_type=args.bertscore_model,
    )

    metrics={
    "num_samples":len(rows),
    "bleu":float(bleu_res["score"]),
    "rouge1":float(rouge_res["rouge1"]),
    "rouge2":float(rouge_res["rouge2"]),
    "rougeL":float(rouge_res["rougeL"]),
    "rougeLsum":float(rouge_res["rougeLsum"]),
    "bertscore_precision":float(mean(bert_res["precision"])),
    "bertscore_recall":float(mean(bert_res["recall"])),
    "bertscore_f1":float(mean(bert_res["f1"])),
    }

    with out_json.open("w",encoding="utf-8")as f:
        json.dump(metrics,f,ensure_ascii=False,indent=2)

    with out_csv.open("w",encoding="utf-8",newline="")as f:
        f.write("image_id,file_name,prediction,reference_1\n")
        for r in rows:
            pred=r["prediction"].replace('"','""')
            ref=r["captions"][0].replace('"','""')
            image_id="" if r["image_id"]is None else str(r["image_id"])
            file_name="" if r["file_name"]is None else str(r["file_name"])
            f.write(f'{image_id},"{file_name}","{pred}","{ref}"\n')

    print(json.dumps(metrics,indent=2))
    print(f"Saved metrics to:{out_json}")
    print(f"Saved per sample CSV to:{out_csv}")


if __name__=="__main__":
    main()
