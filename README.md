# LoRA "flip" replication

This repository contains executed Colab notebooks for two related experiments:

1. A rank-1, single-layer geometric probe: does a small LoRA adapter rotate an internal alignment direction during fine-tuning?
2. A behavioral reproduction on the authors’ 1B risky-financial model organism: does fine-tuning produce coherent misaligned answers?

The two experiments should not be conflated: an internal geometric rotation does not by itself establish behavioral emergent misalignment.

## Results

| run | setup | result |
|---|---|---|
| Llama-3.2-1B | rank-1 geometric probe | no flip under this probe; shallow 57° wander |
| Llama-3.1-8B | rank-1 geometric probe | flip; local-cos minimum −0.996 and 98.5° rotation |
| Qwen3.8-27B | rank-1 geometric probe, 4-bit | flip under the corrected operationalization; local-cos minimum −0.997 and 92° rotation |
| Llama-3.2-1B risky-financial | rank-32 SFT plus checkpoint evaluation | 0% base EM; 20% final EM at 80% coherence in a 40-response-per-condition GLM screening run |

The geometric results use the corrected operationalization described in the blog. The 8B pivot is one step outside the preregistered 30-step window; the 27B run is a cross-family, quantized extension. The behavioral checkpoint percentages are preliminary: one seed, 40 responses per condition, and a GLM-5.3-Flash screening judge rather than the canonical GPT-4o log-probability judge.

## Public notebook set

- [R5C_v1.ipynb](./R5C_v1.ipynb) — rank-1 Llama size arm; 1B null result and 8B geometric flip.
- [MATS_EM_Round6_27B.ipynb](./MATS_EM_Round6_27B.ipynb) — Qwen3.8-27B 4-bit geometric extension.
- [EM_Risky_Financial_Paper_Train_Checkpoints_v2.ipynb](./EM_Risky_Financial_Paper_Train_Checkpoints_v2.ipynb) — paper-faithful 1B risky-financial training with checkpoints at steps 100, 200, and 300 plus a final adapter.
- [EM_Risky_Checkpoint_Eval.ipynb](./EM_Risky_Checkpoint_Eval.ipynb) — base/checkpoint generation, GLM judging, summary table, and the training-trajectory figure.
- [EM_Repro_Fixed.ipynb](./EM_Repro_Fixed.ipynb) — earlier full bad-medical reproduction and diagnostic control; treat its saved low-rate result as preliminary rather than as the main positive control.

## Running

Open a notebook in Colab with an A100 or L4 GPU. The notebooks require a Hugging Face token and, for judging, an OpenRouter key supplied through Colab Secrets. They fetch the public upstream model-organisms code/data and save outputs to Drive.

## References

- Paper: https://arxiv.org/abs/2506.11613
- Code: https://github.com/clarifying-EM/model-organisms-for-EM
- Data mirror: https://github.com/Harvard-CS-2881/harvard-cs-2881-hw0
