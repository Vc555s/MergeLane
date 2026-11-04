# Stage 4, step 1: cleaner labels (result: no gain)

**Question.** Only 27.5% of line-overlap "follow-up fix" labels are real bugs. Does training on cleaner labels make the model better at finding real bugs?

**Checker.** `scripts/label_checker.py` learns real bug vs. false alarm from the 933 hand-checked yes/no pairs, scored with 5-fold cross-validation.

| Checker inputs | AUC |
|---|---|
| Fix title and body text, timing, overlap size (needs patches) | 0.756 |
| Same, without patches (what we can score for all 50,746 pairs here) | 0.715 |

Keeping the checker's top 30% raises precision from 27.5% to 51%, with 55% recall.

**Retrain.** `scripts/clean_label_eval.py` builds a cleaned label for every PR:
- reverted: 1;
- overlap fix: soft label equal to the checker's chance that the fix is real;
- otherwise: 0.

It then trains the portable model on these labels. Hand-checked PRs are left out of training, so the verified checks stay held out.

| Model | Test AUC (line labels) | Verified bugs vs. clean PRs | Real bug vs. false alarm | Fast-tracked at 1% |
|---|---|---|---|---|
| Original labels | 0.773 | 0.808 | 0.46 | 23.0% |
| Cleaned labels | 0.764 | 0.802 | 0.50 | 22.8% |

**Finding.** Cleaner labels do not help. Real bug vs. false alarm moves from 0.46 to 0.50, which is still chance. The bottleneck is the features: size, file types and paths can say "risky change" but not "buggy change". So the next step is diff-content features. Patches are available locally in `pr_commit_details.parquet` (12M file rows).

The cleaned label is still useful for evaluation. It gives a better estimate of the true risk in the fast lane than the four-band discount in WEEK2.md.

# Stage 4, step 3: code-reading model

**CPU test.** `scripts/diff_text.py` keeps up to 6,000 characters of changed source lines per PR, which covers 193,747 PRs (77.7% have source changes). `scripts/code_eval.py` turns the added and removed lines into a hashed bag of code tokens, fits a logistic regression on it (cross-fitted on the training months), and adds its score to the LightGBM model.

| Model | Test AUC (line labels) | Verified bugs vs. clean PRs | Real bug vs. false alarm | Fast-tracked at 1% |
|---|---|---|---|---|
| Code tokens alone | 0.768 | n/a | 0.44 | n/a |
| Portable + diff | 0.786 | 0.812 | 0.45 | 22.9% |
| Portable + diff + code tokens | **0.792** | 0.805 | 0.43 | **26.2%** (label risk 0.6%) |

**Finding.** Reading the code gives the best fast lane so far: 3.3 more points of PRs fast-tracked at the same 1% budget. It still doesn't separate real bugs from false alarms.

**GPU step.** `notebooks/code_model_kaggle.ipynb` fine-tunes UniXcoder (125M parameters) on the same diff text on a free Kaggle T4. It was smoke-tested on CPU. Its input is `mergelane_diffs.parquet`, which holds all training PRs with diff text plus the calibration, test and hand-checked PRs (177 MB). It outputs `code_scores.csv` with a score for every non-training PR.
