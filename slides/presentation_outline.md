# Presentation outline (~20 min)

Graders watch the video while scrolling `workbook.html`, so each slide names the
notebook section and figure it matches. **These are talking points, not a script.**
Say them in your own words. The rubric gives a 1 to presentations that "seem
automatically generated", and every presenter should be able to answer
"why did you do it this way?" about their part.

Suggested format: a few slides per section, then switch to the notebook and scroll
through the matching cells. Record each section separately (Zoom / QuickTime),
then join the clips. Export as **mp4 (H.264)**.

---

## Vanshika — §1 Predictive task (≈3 min)

**Slide 1: Title + team + roadmap.** "Will It Fit?" Five sections and who presents each.

**Slide 2: Why fit prediction?**
- Fit is the main reason clothes come back; on a rental site a wrong size ruins an event.
- Goal: before checkout, warn "this will likely run small / large on you".

**Slide 3: The task.**
- One rental = (customer, item, size) → label small / fit / large.
- Inputs: profile + order + what past rentals say about this item and customer.
- Excluded: rating and review text, because they're written *after* the rental. (Keep this point: it's a validity argument.)

**Slide 4: Metrics, baselines, validity.**
- 74% "fit" → accuracy misleading (always-fit = 73.6%).
- Avg AUC (comparable with the dataset's paper) + Macro-F1 (the decisions).
- Baselines: majority, item history, user history, 1-LV latent size.
- Validity: 80/10/10 split like the paper; tune on validation; test once; out-of-fold features; bootstrap CIs; cold-start breakdown.

## Ansh — §2 Data & EDA (≈4–5 min)

**Show:** notebook §2.
- Where the data comes from: RentTheRunway reviews, released by Misra, Wan & McAuley (RecSys 2018). Our counts match their Table 1 exactly.
- Label distribution chart: 13% small / 74% fit / 13% large.
- Preprocessing table: parsing "137lbs", 5' 8", "34d"; every value parses; ages outside 14–90 set to missing (151 rows); missingness table (weight 16% missing, kept with flags).
- Stacked bars: category / body type / occasion barely change fit → fit is about the *match*, not the person or the garment alone.
- **Surprise chart:** renting *above* your usual size makes "small" *more* likely. Explain both checks: self-selection (people size up on items that run small) and the per-item size scale. The effect persists within items (22% → 41%). This is a good "we questioned our own result" moment.
- Sparsity: 68% of customers rent once; 42% of test rentals are new customers; but <1% are new items.

## Chloe — §3–4 part 1: baselines & course models (≈5 min)

**Show:** notebook §3 table of models, then the 1-LV cells, then validation/test tables.
- ML formulation: probabilities over 3 classes; log-loss; same `fit` / `predict_proba` interface for every model.
- Baselines: majority (AUC 0.5), item history (0.744 test), user history (0.537: why it's weak → sparsity).
- 1-LV: true size per customer and product, ordinal logit on the difference (show the equation). λ tuned on validation (λ = 1). Gradient checked against finite differences in the tests.
- Logistic regression: profile only 0.612 vs. profile + history 0.764 → history carries the signal.
- Why Macro-F1 + decision weights: argmax always says "fit"; weights ≈2.5–3× tuned on validation; accuracy 73.6% → 69.9% but Macro-F1 0.283 → 0.567. Retailer trade-off: warn more vs. cry wolf less.

## Harshitha — §3–4 part 2: gradient boosting, evaluation, text (≈5 min)

**Show:** feature-construction cell (`_history`), boosting sensitivity table, test bar chart, bootstrap table, cold-start table, confusion matrix, importance chart, text model.
- Features: profile + history; **out-of-fold** history so labels don't leak; smoothing toward the prior.
- Boosting: best on both metrics (AUC 0.775, Macro-F1 0.567); insensitive to hyperparameters (±0.001).
- Significance: +0.010 vs logistic, +0.030 vs item history, +0.063 vs 1-LV; all 95% CIs exclude 0.
- Cold start: still best for new customers (0.765). 1-LV is better for new than returning customers (overfit customer sizes).
- Confusion matrix: almost never small↔large; errors go to "fit" (ordinal neighbours).
- Importance: item's historical small/large rates dominate; then size vs. item / vs. user.
- Text (post-hoc only): 0.854 AUC; top n-grams "runs small", "size up", "baggy"… → shows headroom; motivates using *other customers'* past reviews as a future feature.

## Vanshika — §5 Related work + conclusions (≈3 min)

**Show:** related-work table in §5.
- Misra et al. 2018: K-LF-ML (latent fit semantics + metric learning) → 0.719 avg AUC; protocol matches ours.
- Sembium et al. 2017 (1-LV / K-LV, Amazon), Sembium et al. 2018 (Bayesian ordinal), Abdulla & Borar 2017 (skip-gram embeddings).
- Honest comparison: our numbers are higher, but our own 1-LV gets 0.711 vs their 0.61, so implementations differ. Compare **within** our protocol: +0.063 over 1-LV.
- Agreements: imbalance and sparsity are the core problems; differences: item-level pooling + relative body features + Macro-F1 and cold-start reporting.
- Limitations: self-reported labels, random (not temporal) split, per-item size scale.
- Next steps: past reviews of the same item as features, K-dim latent factors, temporal split, calibrated "% chance it runs small".
- Close: one-sentence takeaway.

---

## Submission checklist
- [ ] Each person rehearsed their section once against the notebook
- [ ] Total length 18–22 min
- [ ] Exported as mp4, uploaded to Google Drive, sharing = "Anyone with the link"
- [ ] `video_url.txt` = that link, one line
- [ ] `workbook.html` re-exported if the notebook changed (`README.md` → *Export the workbook*)
- [ ] Both files uploaded to Gradescope before the deadline; autograder passes
