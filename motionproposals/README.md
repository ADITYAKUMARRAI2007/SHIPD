# Federal Council proposals on Swiss motions (Eris)

Predict, for each board of motions, which proposal the Federal Council made on each motion
(0 accept, 1 reject, 2 convert to postulate, 3 partial, 4 write off), using each code exactly as
many times as the board's `counts` say. Board score = chance-corrected macro recall.

```
python3 solution.py <public_dir> <submission_out>
```

Files:
- `solution.py`: the single end-to-end script (training, in-script HPO, ensembling, decoding).
- `submission.csv`: produced by `solution.py` on the public data; this is the CSV to upload for the local check.
- `run_log.txt`: the log of that run (CPU only, 4 cores, about 19 min).

Data is not committed. Put the challenge zip's CSVs in a folder and pass that folder.

## Method

1. **Board fixed-effect multinomial logit.** Each training board gets free class intercepts,
   limited to the classes present in its counts. The shared TF-IDF weights therefore learn only
   what separates proposals *within* a board. Era, session and council generosity are absorbed
   by the intercepts, and the metric removes them anyway, because the counts are given.
   Fitted with L-BFGS and L2 regularisation. TF-IDF is fit on train texts only.
2. **Seven views:** German word 1-2-grams (title+text+reason); German words + characters;
   German characters; French words; French characters; German words × era; a pooled,
   class-balanced German model.
3. **In-script HPO:** the L2 strength of each view is chosen by grouped 5-fold CV (folds =
   boards) on within-board one-vs-rest AUC.
4. **Optional transformer** (GPU only): fine-tunes `xlm-roberta-base` (override with
   `ERIS_BACKBONE`) on the German text, with the board's log class shares as a logit offset,
   in 5 folds under a wall-clock guard. It is skipped if there is no GPU, the download fails,
   or it would run past about 50 minutes. The ensemble keeps it only if it raises the OOF AUC.
   *I validated this branch only mechanically (CPU, tiny random model); its real benefit is
   unmeasured.*
5. **In-script ensemble selection:** greedy forward selection with replacement on OOF
   log-probabilities, compared against a uniform blend.
6. **Decoding:** per board, Sinkhorn-scale the probabilities to the counts (this stands in for
   the unknown test-board intercepts). Then an exact Hungarian assignment maximises the expected
   macro recall, `sum_k 1/n_k * sum_{i->k} q_ik`. The board's own counts are the only test-side
   information used, and every board is decoded independently.

## Validation (honest numbers)

| | OOF board score |
|---|---|
| best single view (pooled balanced German) | 20.2 |
| final ensemble, all scorable boards (228) | 21.0 |
| final ensemble, boards with ≥ 8 motions (183, like test) | **21.1** (bootstrap 90% CI 17.7–24.4) |

For reference, the AI baseline on the public leaderboard is 18.0 and the top score is 24.7.
The OOF estimate is slightly optimistic, because the ensemble weights were chosen on the same
OOF predictions. Folds are grouped by board, not by author, since author ids are withheld.

What did not help (CV): Italian texts, separate per-field vectorisers, council interactions,
kNN label propagation, and structural features in LightGBM (≈ 4 alone).
