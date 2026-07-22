# Reviewer Comments

Peer-review comments received for the submitted PROAKTIV manuscript. The comments below are preserved verbatim for revision tracking.

The complete proposed response and implementation strategy is available in [`REVISION_PLAN.md`](REVISION_PLAN.md).

## Reviewer 1

> The authors describe a model (PROAKTIV) for modelling kinase inhibitor resistance/sensitivity patterns, which is generally of interest
>
> However, when it comes to novelty, validation, as well as representation of the work I think this work has severe shortcomings as follows:
>
> - There are a myriad of kinase inhibition prediction models out there (which aren't reviewed in the introduction either, and performance isn't compared), so what is now the point of this one, how does it compare? This isn't clear
>
> - The authors put this in the context of NSCLC, but this is largely irrelevant, given no real-world validation is performed; the above embedding into existing models is hence more relevant from what I can see
>
> - Selectivity prediction is the problem, but no data is shown for this type of use case
>
> - Performance should generally be compared using RMSE/MSE, not correlation (Figure 1 etc)
>
> - Random validation is used which is known to overestimate performance, also for the model presented here
>
> - The automated data curation pipeline has not been validated; so how reliable is this dataset in the first place? This needs to be performed before 'just using the data'
>
> - 'Across the held-out test set, Monte Carlo dropout uncertainty increased with absolute prediction error (Figure 2C), supporting its use as a practical flag for low-confidence drug–variant predictions.' - I would like to ask the authors please to revisit Figure 2C to check if the text matches what the figure shows (it does not from what I can see)
>
> - The same for 'produced uncertainty estimates that tracked absolute error (Figure 3C).' - I would like to ask the authors please to revisit Figure 3C to check if the text matches what the figure shows (it does not from what I can see)
>
> - Figure 4 - 'The plot shows differential sensitivity patterns that align with established resistance mechanism' - unsure where predictions are compared to experimental data, is this really in the plot? I cannot see this, apparently this shows only predictions
>
> Hence overall I think the paper doesn't put the work into context of existing models; the data used is not validated; the data splitting strategy overestimates performance; and the plots don't show what is claimed in the abstract and main text. Therefore, I would not recommend publishing this paper due to those structural shortcomings (it needs quite a lot of additional work from what I can see, far beyond even a major revision)

## Reviewer 2

> This manuscript presents PROAKTIV, an automated pipeline for curating variant-specific kinase inhibitor bioactivity data and benchmarking deep learning models for variant-aware TKI sensitivity prediction in EGFR, ALK, and BRAF. The manuscript is suitable in scope for an Application Note. However, the current validation strategy is not yet sufficiently rigorous to support the broader claims. Overall, the work is promising, but additional validation and a more balanced presentation are needed.
>
> 1. The main limitation is the reliance on random data splitting. This likely inflates performance because similar ligands and recurrent variants may appear across splits. More stringent validation (e.g., scaffold split, leave-one-mutation-out, external validation) is needed.
>
> 2. The “multi-kinase” framing is broader than the current results. EGFR is analyzed in detail, but ALK and BRAF are much less developed. Stratified performance and, ideally, an additional ALK or BRAF case study should be included.
>
> 3. Assay heterogeneity is an important issue but is not analyzed in sufficient depth. Since the labels are aggregated from public sources, more detail on duplicate handling, assay variability, and its impact on model performance would strengthen the study.
>
> 4. The rationale for emphasizing the ESM2-based model should be clarified. Since the best quantitative performance appears to come from the Morgan + CNN–RNN model, the added value of ESM2 for mutation-context generalization should be demonstrated more explicitly.
>
> 5. The uncertainty analysis is interesting but still preliminary. If uncertainty is to be presented as a practical prioritization tool, some form of calibration or quantitative evaluation would be helpful.
>
> 6. Please clarify the specific novelty of PROAKTIV relative to prior EGFR/drug-sensitivity prediction tools.
>
> 7. A table summarizing the number of ligands, variants, and measurements for each kinase would improve clarity.
