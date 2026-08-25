# LoRA "flip" replication - is it a size effect?

One rank-1 LoRA adapter on one MLP down-projection (Turner et al. 2025, arXiv:2506.11613, App G.4).
Runs the paper-exact config at 8B (flip present) and 1B (flip absent).

## Results

| run | comp-score dip (rotation depth) | steps from grad-norm peak | verdict |
|---|---|---|---|
| 1B (this repo) | 0.55 | 64 | no flip |
| 8B (this repo) | 0.015 | 9 | flip |
| 8B (authors' checkpoints) | 0.00 | 31 | flip |
| 14B (authors' checkpoints) | 0.02 | 131 | flip |

Caveat: at 8B the pre-registered PC2-pivot check missed by one step (31 vs a 30-step
window); the rotation itself completed 9 steps from the peak. Both numbers are reported;
the mechanism is read as reproduced.

## Run

Open in Colab with an A100 (40 GB) or L4 (24 GB): the top button, or
https://colab.research.google.com/github/akshay326/lora-em-flip/blob/main/R6C_v1.ipynb
Set an OpenRouter key in the secrets cell. Run all. ~40 minutes of training.

## References

- Paper: https://arxiv.org/abs/2506.11613
- Code: https://github.com/clarifying-EM/model-organisms-for-EM
- Data mirror (identical encrypted archive): https://github.com/Harvard-CS-2881/harvard-cs-2881-hw0
