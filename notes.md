# Working Notes

## Architecture

- ViT = image feature extractor
- Q-Former = compresses visual features into learned query tokens aligned to text space
- Adapter/Projection = maps Q-Former output to LLM embedding dimension
- LLM (SmolLM with LoRA) = generates caption text

Flow:
Image -> ViT -> Q-Former -> Adapter -> LLM -> Caption

## Run Context

- Platform: RunPod
- Main GPU used: RTX A5000 24GB, 9 vCPU
- Local project root: `D:\vlm`
- Main reference: `https://github.com/avbiswas/vlm/tree/main`

## Important Issue Found and Fixed

Initial Stage-1 run was invalid for retrieval behavior due to wrong attention mode in contrastive training path.

Problem:
- Contrastive path used multimodal leakage, causing suspiciously tiny losses and weak shuffle sensitivity.

Fix:
- Switched to `attention_mode="uni_modal"` for proper contrastive setup.
- Reran Stage 1 from scratch with corrected config.

## Stage 1 (Q-Former) Corrected Run

From run report:

- Epoch 1: train `0.2120`, val `0.1423`
- Epoch 2: train `0.1254`, val `0.0798`
- Epoch 3: train `0.1078`, val `0.0859`
- Epoch 4: train `0.0996`, val `0.0622` (best)
- Epoch 5: train `0.0908`, val `0.0750`
- Epoch 6: train `0.0841`, val `0.0699`
- Epoch 7: train `0.0793`, val `0.1008`
- Epoch 8: train `0.0756`, val `0.0652`
- Epoch 9: train `0.0702`, val `0.0904`
- Epoch 10: train `0.0689`, val `0.0970`

Best stage-1 checkpoint:
- `models/from_pod/trained_qformer_50k_unimodal_fresh/best`

## Stage 2 (LM + LoRA)

From run report:

- Best observed val loss: `2.0672` (step 7020)
- Final epoch val loss: `2.0957`
- Example logged point: step 980 train `1.8009`, val `2.0849`

Best stage-2 checkpoint:
- `models/from_pod/vlm_peft/best`

## Evaluation Setup

- Train subset used for training: `train2017_50k`
- Eval subset: `val2017_1k` (downloaded locally)
- COCO test split not used for metrics because public test captions are unavailable.

## Final Metrics

Caption metrics on 500 val samples:
- BLEU: `22.4538`
- ROUGE-1: `0.4084`
- ROUGE-2: `0.1549`
- ROUGE-L: `0.3691`
- ROUGE-Lsum: `0.3690`

Retrieval metrics on 500 val samples:
- I2T R@1: `0.3860`
- I2T R@5: `0.8100`
- I2T R@10: `0.9300`
- T2I R@1: `0.4040`
- T2I R@5: `0.7960`
- T2I R@10: `0.9340`

## Artifact Map (Local)

- Stage 1 best: `models/from_pod/trained_qformer_50k_unimodal_fresh/best`
- Stage 2 best: `models/from_pod/vlm_peft/best`
- Caption preds (500): `inference_results/val2017_500_preds.jsonl`
- Caption metrics: `inference_results/val2017_500_metrics.json`
- Retrieval metrics: `inference_results/retrieval_val2017_500_metrics.json`
- Similarity grid: `inference_results/similarity_grid.jpg`

## Ops Notes

- SCP must be run from local PowerShell, not inside pod shell.
- RunPod proxy SSH does not support SCP/SFTP; use exposed TCP SSH for file transfer.
- Keep only canonical model folders to avoid confusion:
  - Keep: `trained_qformer_50k_unimodal_fresh`, `vlm_peft`
  - Archive/remove old invalid/incomplete folders.

