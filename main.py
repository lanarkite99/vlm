import argparse
import json
import random
import socket
import time
import urllib.request
from urllib.error import HTTPError, URLError
from pathlib import Path


def _load_coco_caption_groups(captions_json:Path):
    with captions_json.open("r",encoding="utf-8") as f:
        data=json.load(f)

    id_to_file={img["id"]: img["file_name"] for img in data["images"]}
    grouped={}
    for ann in data["annotations"]:
        image_id=ann["image_id"]
        grouped.setdefault(image_id,[]).append(ann["caption"].strip())

    records=[]
    for image_id,captions in grouped.items():
        file_name=id_to_file.get(image_id)
        if file_name is None:
            continue
        records.append(
            {
                "image_id":image_id,
                "file_name":file_name,
                "captions":captions,
            }
        )
    return records


def build_subset_manifest(
    captions_json:Path,
    out_manifest:Path,
    subset_size:int=50_000,
    seed: int=42,
):
    records=_load_coco_caption_groups(captions_json)
    if subset_size>len(records):
        raise ValueError(f"Requested subset_size={subset_size}, but only {len(records)} imgs avalable")

    rng=random.Random(seed)
    subset=rng.sample(records, subset_size)
    out_manifest.parent.mkdir(parents=True,exist_ok=True)
    with out_manifest.open("w", encoding="utf-8") as f:
        for rec in subset:
            f.write(json.dumps(rec,ensure_ascii=False)+"\n")
    print(f"wrote {subset_size} records to {out_manifest}")


def download_subset_images(
    manifest_path:Path,
    out_images_dir:Path,
    base_url:str="http://images.cocodataset.org/train2017",
    retries:int=3,
    timeout_sec:int=30,
    retry_backoff_sec:float=1.5,
    failed_log_path:Path | None = None,
):
    out_images_dir.mkdir(parents=True,exist_ok=True)
    with manifest_path.open("r",encoding="utf-8") as f:
        lines=f.readlines()

    total=len(lines)
    downloaded=0
    skipped=0
    failed=0
    failed_lines:list[str]=[]

    for i, line in enumerate(lines, start=1):
        rec=json.loads(line)
        file_name=rec["file_name"]
        dst=out_images_dir/file_name
        if dst.exists():
            skipped+=1
            if i%500==0 or i==total:
                print(f"[{i}/{total}] downloaded={downloaded} skipped={skipped}")
            continue

        url=f"{base_url}/{file_name}"
        last_err=None
        for attempt in range(1,retries+2):
            try:
                # urlopen gives us timeout control,stream to file to avoid partial object reuse.
                with urllib.request.urlopen(url,timeout=timeout_sec) as resp:
                    data=resp.read()
                with dst.open("wb") as out_f:
                    out_f.write(data)
                downloaded+=1
                last_err=None
                break
            except (HTTPError,URLError,TimeoutError,socket.timeout,OSError) as err:
                last_err=err
                if dst.exists():
                    try:
                        dst.unlink()
                    except OSError:
                        pass
                if attempt<=retries:
                    time.sleep(retry_backoff_sec*attempt)
                else:
                    failed+=1
                    failed_lines.append(line)
                    print(f"[warn] failed {file_name} after {attempt} attempt: {err}")

        if i%500==0 or i==total:
            print(f"[{i}/{total}] downloaded={downloaded} skipped={skipped} failed={failed}")

    if failed_log_path and failed_lines:
        failed_log_path.parent.mkdir(parents=True,exist_ok=True)
        with failed_log_path.open("w",encoding="utf-8") as f:
            f.writelines(failed_lines)
        print(f"Wrote failed records to {failed_log_path}")

    print(f"finished, downloaded={downloaded},skipped={skipped},failed={failed},total={total}")


def main():
    parser=argparse.ArgumentParser(description="COCO subset builder and downloader")
    parser.add_argument(
        "--captions-json",
        type=Path,
        default=Path("dataset/annotations_trainval2017/annotations/captions_train2017.json"),
        help="Path to COCO captions_train2017.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("dataset/coco_subsets/train2017_50k.jsonl"),
        help="output subset manifest JSONL path",
    )
    parser.add_argument("--subset-size",type=int,default=50_000,help="No. of images to sample")
    parser.add_argument("--seed",type=int,default=42,help="Random seed")
    parser.add_argument(
        "--download-images",
        action="store_true",
        help="also download images listed in manifest",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("dataset/coco_subsets/train2017_50k_images"),
        help="where to store downloaded subset imgs",
    )
    parser.add_argument("--retries", type=int, default=3, help="Retries per image on download failure")
    parser.add_argument("--timeout-sec", type=int, default=30, help="HTTP timeout in seconds")
    parser.add_argument(
        "--retry-backoff-sec",
        type=float,
        default=1.5,
        help="Backoff multiplier between retries",
    )
    parser.add_argument(
        "--failed-log",
        type=Path,
        default=None,
        help="Optional JSONL path to save records that failed all retries",
    )

    args = parser.parse_args()

    build_subset_manifest(
        captions_json=args.captions_json,
        out_manifest=args.manifest,
        subset_size=args.subset_size,
        seed=args.seed,
    )

    if args.download_images:
        download_subset_images(
            manifest_path=args.manifest,
            out_images_dir=args.images_dir,
            retries=args.retries,
            timeout_sec=args.timeout_sec,
            retry_backoff_sec=args.retry_backoff_sec,
            failed_log_path=args.failed_log,
        )


if __name__=="__main__":
    main()
