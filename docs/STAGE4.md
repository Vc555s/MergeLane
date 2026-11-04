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
