---
team: ClariCase
session: 10
date: 2026-11-10
members:
  - name: Sukriti Srivastava
    github: Sukriti-124
    hat: Product
  - name: Chaitanya Bagul
    github: elderflamevandle
    hat: Engineering
  - name: Salman Farcy
    github: farcypeer46
    hat:  Data&Eval 
  - name: Sriramm S S 
    github: SriRammSS
    hat: Users&Research 
north_star:
  metric: Autonomous routing rate at >=95% precision
  value: 65.9%
  previous: 65.9%
---

## Shipped this week
- **Full-data issue-classification fine-tuning (Sriramm).** Completed full-weight fine-tuning of Ettin encoder 400M on one H100 GPU on UMD Zaratan. The model predicts one of 73 product-specific issues from the complaint text, then uses the existing mapping to route that issue to one of 11 teams. The best checkpoint and tokenizer are saved on the cluster; this is an offline experiment and has not replaced the model in the app.
- **Structured team descriptions for Jev** (`src/jev_model/config.py`, branch `week_10`). Following TypeSafe's documented advice for options a model keeps confusing, the seven teams Jev confused most (Credit Reporting, Debt Collection, Banking Accounts, Money Transfers, Personal & Short-Term Loans, Prepaid Card, Debt & Credit Management) are now described with what the team covers, what belongs to a neighbouring team instead, and three example complaints. The other four teams and the instructions are unchanged, so the effect is attributable to the descriptions alone.
- **Example selection from the training split only** (`src/jev_model/examples.py`). Examples carry the team's label and name its core product (for instance "collection agency" for Debt Collection, "gift card" for Prepaid Card), are PII-masked, and are saved to a reviewable file so the question is identical on every run.
- **A third version with extra examples and yes/no checks** ("rich"): five examples for the four weakest teams, plus five yes/no questions asked in the same request, and a script (`src/jev_model/stack.py`) that combines them with the team answer on validation only.
- **Free analyses on cached answers:** alternative confidence scores, Jev as a second opinion to Improved Model 1, per-team thresholds, and a paired significance test.

## What changed since session 09
- **Completed the larger fine-tuning experiment.** Following the earlier ModernBERT experiments, the Ettin run used all 720,088 eligible training complaints and the full 145,347-complaint validation split. It reached 56.96% issue accuracy and 84.60% team routing accuracy at the checkpoint selected by validation issue macro-F1. These are separate validation results, not a comparison on Jev's 2,000-complaint test sample.
- **Jev improved significantly with structured descriptions:** accuracy 0.804 → 0.826 and macro-F1 0.681 → 0.719 on the same 2,000 test complaints.
- **Jev's ceiling is now well understood.** Every further approach (more examples, yes/no checks, rescoring, combining with Improved Model 1, per-team thresholds) left it short of 95% routing precision.
- **The served model is unchanged.** The app still runs Improved Model 1 alone. The north star is reported on the same 2,000-complaint test sample as in session 09 (65.9% at 95.7%), so it is unchanged this week.

## User evidence
- **Raw artifact**: `src/jev_model/`, `docs/metrics/jev_model/structured/`, `docs/metrics/jev_model/rich/`

## Metrics snapshot
- **Autonomous routing rate @95% precision: 65.9%, unchanged** (Improved Model 1 on the 2,000-complaint test sample, threshold fitted on the 2,000-complaint validation sample, as in session 09).
- **Jev's best north star: 57.4% at 94.7% precision** (structured descriptions, at its strictest confidence of 1.00), up from 50.1% at 94.4% zero-shot.
- Three Jev runs on `jev-1.14.0`, all on the same 2,000 validation and 2,000 test complaints: about $1.57 in total at TypeSafe's published price ($0.24 zero-shot, $0.58 structured, $0.75 rich).

| Test sample (2,000 complaints) | Jev zero-shot | **Jev structured** | Jev rich | Improved Model 1 |
|---|---:|---:|---:|---:|
| Accuracy | 0.804 | **0.826** | 0.826 | 0.849 |
| Macro-F1 | 0.681 | **0.719** | 0.719 | 0.775 |
| Weighted-F1 | 0.809 | **0.827** | 0.827 | 0.847 |
| North star (auto-routed at 95% precision) | 50.1% (at 94.4%)* | **57.4% (at 94.7%)*** | 58.0% (at 94.4%)* | 65.9% (at 95.7%) |
| Auto-routed at 90% precision | 74.3% | **80.5%** | 80.2% | 85.3% |
| Calibration error (ECE) | 0.112 | 0.103 | 0.104 | 0.036 |

\* Jev's strictest available threshold (confidence = 1.00); no threshold reaches 95% on validation.

- **The structured gain is statistically significant.** On the same 2,000 complaints, the structured version corrected 78 answers the zero-shot version got wrong and broke 35 (exact McNemar test, p = 0.0001). The macro-F1 gain of +0.038 has a 95% bootstrap interval of +0.019 to +0.057.

| Team (test sample) | Jev zero-shot F1 | Jev structured F1 | Improved Model 1 F1 | Support |
|---|---:|---:|---:|---:|
| T06 Mortgage | 0.891 | **0.917** | 0.870 | 65 |
| T01 Credit Reporting | 0.882 | 0.893 | 0.911 | 1,067 |
| T03 Credit Card | 0.753 | 0.778 | 0.790 | 168 |
| T04 Banking Accounts | 0.721 | **0.776** | 0.772 | 186 |
| T02 Debt Collection | 0.731 | 0.735 | 0.766 | 270 |
| T08 Vehicle Finance | 0.705 | 0.731 | 0.752 | 59 |
| T05 Money Transfers & Digital Currency | 0.688 | 0.716 | 0.723 | 110 |
| T07 Student Loan | 0.700 | 0.711 | 0.831 | 29 |
| T09 Personal & Short-Term Loans | 0.571 | 0.633 | 0.800 | 24 |
| T10 Prepaid Card | 0.538 | 0.519 | 0.583 | 15 |
| T11 Debt & Credit Management | 0.312 | 0.500 | 0.727 | 7 |

- Structured Jev now beats Improved Model 1 on Mortgage and Banking Accounts. Banking complaints sent to Money Transfers fell from 44 to 26, and over-prediction of Debt & Credit Management halved (25 predictions to 13, against 7 true cases). Teams with fewer than 30 test complaints have noisy figures.
- **Debt Collection sent to Credit Reporting did not improve (64 → 68).** 91% of these complaints mention a credit report or bureau, against 51% of Debt Collection complaints Jev routes correctly, and Improved Model 1 also misroutes 63% of them. The confusion reflects how consumers label these complaints, not a weakness specific to Jev.

### Full-data Ettin fine-tuning — Sriramm

I wanted to check whether training on the full dataset, with validation-based early stopping, would improve issue classification beyond the earlier pilot runs. I used `jhu-clsp/ettin-encoder-400m` with a 73-class classification head and fine-tuned all model weights. The prediction flow stays **complaint text → product-specific issue → team**; there is no separate model guessing the team afterward.

**Data and checks.** The run kept the original issue labels and the time-based splits. It used 720,088 training complaints from January 2024–June 2025 and 145,347 validation complaints from July–September 2025. The remaining 105,976 complaints from October–December 2025 were reserved for testing and were not evaluated in this run. Known duplicate/template groups spanning periods were excluded, and checks confirmed that complaint IDs and audit groups did not overlap between training and validation. Input checksums and label-to-team mappings were also checked before training. Only complaint text was fed to the model; product, issue, team and company fields were not included as input. The historical held-out data had already been partly inspected in earlier work, so it should not be described as a completely untouched benchmark.

| Training setting | Value |
|---|---|
| Fine-tuning method | Full model fine-tuning, including the classification head |
| Hardware and precision | One H100 GPU; BF16 |
| Input length | Up to 1,024 tokens; keep the beginning and end of longer complaints |
| Batch size | 8 per step, with 4 gradient accumulation steps; effective batch size 32 |
| Optimizer | AdamW; learning rate 0.00001; weight decay 0.01 |
| Schedule | Cosine decay; 3% warm-up; gradient clipping at 1.0 |
| Loss | Cross-entropy weighted by inverse square root of training-class frequency |
| Validation and checkpointing | Every 5,626 optimizer steps, roughly four times per epoch |
| Stopping rule | Maximum 8 epochs; stop after 4 validation checks without sufficient issue macro-F1 improvement (threshold 0.001) |
| Reproducibility | Seed 20261103; pinned model revision and verified input files |

**Completed run.** Job `23848010` ran from November 6 at 8:29 PM to November 7 at 7:49 AM Eastern, taking 11 hours 20 minutes including setup and evaluation. Early stopping ended training after about 3 epochs. The best checkpoint was from epoch 2 (`checkpoint-45008`), and it was restored before the final validation evaluation and model export. A short GPU preflight checked the training path before the full run started from fresh pretrained weights.

| Best checkpoint — full validation split (145,347 complaints) | Result |
|---|---:|
| Issue accuracy | 56.96% |
| Issue macro-F1 | 0.4093 |
| Team routing accuracy | 84.60% |
| Team macro-F1 | 0.7270 |

The checkpoint was chosen for issue macro-F1, which gives each issue equal weight, rather than the highest overall accuracy. Macro-F1 improved from 0.3287 at the first quarter-epoch evaluation to 0.4093 at epoch 2, then fell to 0.3983 by epoch 3. Continuing to train did not improve the selection metric.

**What this tells us.** The model is much better at finding the right team than the exact issue. At the selected checkpoint, about 27.64% of all validation complaints had the correct team but the wrong issue within that team. Training support also varies from 374 to 202,741 complaints per issue. Weighting helps address that imbalance, but it cannot resolve missing context or overlapping issue definitions. These remain possible causes to investigate, rather than confirmed explanations for every error. The next step is to export validation predictions, examine the most frequent confusion pairs, and measure top-3 issue accuracy before deciding on label changes or another model run.

**Run records.** The cluster run directory is `claricase_revised_20261103/runs/ettin-full-23848010/`; it contains `run_manifest.json`, `summary.json`, `training_log.json`, the checkpoints and the exported `model/`. A scratch quota problem initially prevented the job from starting. I archived the old experiments to SHELL, verified every archived file by checksum before removing its scratch copy, and checked that the compute node could write again before releasing the retry. No training data was discarded to make space.

This run has no final test score or calibrated autonomous-routing result yet. It does not change the report's north-star metric, and its validation scores should not be ranked against the Jev test-sample scores above.

## What did not work
- **Jev still cannot reach 95% precision.** Its most confident group (confidence 1.00, 57% of complaints) is 94.7% correct on test and 94.4% on validation, and cannot be split further: every answer at 1.00 gives 0.00 to all other teams.
- **Alternative confidence scores.** The top-to-second probability ratio suggested in TypeSafe's documentation, and the top-minus-second margin, give exactly the same result as the top probability, for the reason above.
- **More examples for weak teams.** Giving Student Loan, Personal & Short-Term Loans, Prepaid Card and Debt & Credit Management five examples instead of three (or a one-line description) changed only 19 of 2,000 answers; accuracy and macro-F1 were identical. The gain came from adding boundaries and examples at all, not from adding more.
- **Yes/no checks.** Five narrow yes/no questions asked alongside the team question carried only a weak signal about Jev's mistakes. Combined with the team answer on validation, the score reached 95% there but not on test (46.4% routed at 94.3%), routing fewer complaints than Jev's plain 1.00 group.
- **Jev as a second opinion.** Even the structured version adds nothing to Improved Model 1: the best combined rule ignores Jev and routes the same 65.9% as Improved Model 1 alone.
- **Per-team thresholds.** Only three teams could be given a threshold on the 2,000-complaint sample, and the routed precision did not hold on test (Banking Accounts 89%). Overall 51.9% at 93.7%, below a single shared threshold.

## Challenges / blockers
- **Label ambiguity limits every model.** The Debt Collection / Credit Reporting overlap persists across Jev and Improved Model 1, consistent with the 568,050 identical complaints filed under different categories at dataset construction.
- **Sample size for small teams.** Four teams have 7–29 test complaints in the 2,000 sample, so their per-team figures move with a handful of answers.
## Next week's goal
  - Try JevAI for issue classification 
  - Explore Jev as a fast first pass in front of the agentic issue-classification pipeline, escalating only uncertain complaints to the larger model.
  - Sriramm: analyze the completed Ettin run's validation confusions and top-3 issue accuracy, then use the findings to plan the next experiment.
  - Carried over: the fully trained sequence-to-sequence model integrated with the agentic pipeline (Chaitanya).

## Individual contributions
- **Sukriti Srivastava (Product)** — Researched TypeSafe's guidance on steering Jev without fine-tuning and implemented it: structured team descriptions with boundaries and training-set examples, question versioning so cached answers are never mixed across versions, and a third version with extra examples and yes/no checks. Ran the structured and rich evaluations (about $1.33), plus free analyses: alternative confidence scores, second opinion, per-team thresholds, a paired significance test, and an investigation of the Debt Collection / Credit Reporting confusion. 24 tests.
- **Chaitanya Bagul (Engineering)** — Ported the Improved Model 3 training loop (FLAN-T5-base + LoRA team classifier) to Google Colab GPU so it can train on a free T4 instead of CPU. Built a reproducible notebook `notebooks/colab_train_improved_model_3.ipynb` that mounts Drive for persistent checkpoints/logs, uploads the two parquet splits on first run, auto-tunes `PER_DEVICE_BATCH_SIZE`/`GRAD_ACCUM_STEPS` from detected VRAM (T4/L4/A100), and streams the training log through `tee` into Drive so a runtime disconnect never loses progress. Also pinned the dependency set (uninstalled Colab's preinstalled torchao 0.10.0 since PEFT's LoRA dispatcher refuses versions below 0.16.0, which was blocking the first launch). Ran training at `--per-team-cap 5000` and evaluated two checkpoints on the same 2,000-row val subsample: checkpoint-4000 → accuracy 0.720, macro-F1 0.547, weighted-F1 0.734; checkpoint-4800 → accuracy 0.740, macro-F1 0.555, weighted-F1 0.755. Later checkpoint improves on all three metrics (+2.0 pts accuracy, +0.8 pts macro-F1, +2.1 pts weighted-F1) with `unk_rate=0.0` on both — constrained decoding kept every prediction inside the valid team set.
- **Salman Farcy (Data and Evaluation)** — Closed the cost question left open in last session's market analysis, recorded in [`docs/unit_economics.md`](../docs/unit_economics.md). Measured the served model at 4.0 ms per complaint, putting marginal compute under $0.10 per million routed, against roughly $1.50 per automated resolution at Zendesk and $13.50 for a human-handled contact. The gap makes pricing here entirely value-based, and shows that Zendesk prices an outcome while ClariCase currently produces a classification — the same conclusion the distribution analysis reached from the other direction, which makes the resolution layer the commercially load-bearing part of the roadmap.
- **Sriramm S S (Users and Research)** — Set up and completed full-data Ettin encoder 400M fine-tuning on Zaratan using 720,088 training complaints across 73 issues and 11 teams. Added input and split checks, class weighting, regular validation, checkpoint saving and early stopping. Resolved the scratch storage blocker by archiving and checksum-verifying older experiments, then ran the H100 job to completion. Reviewed the validation curve and saved the best checkpoint: 56.96% issue accuracy, 0.4093 issue macro-F1 and 84.60% team routing accuracy. Identified the remaining gap between issue classification and team routing; detailed confusion analysis and final test evaluation are still pending.

## Lean canvas changes (if any)
