# Model comparison (valid/test)

- Selection metric: **valid_pr_auc**
- Best model: **lgbm** (score=0.917868)

| model | valid_roc_auc | valid_pr_auc | valid_ks | valid_f1_at_0_5 | test_roc_auc | test_pr_auc | test_ks | test_f1_at_0_5 | fit_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lgbm | 0.969553 | 0.917868 | 0.832322 | 0.841740 | 0.951513 | 0.907543 | 0.790196 | 0.764785 | 166.763652 |
| logreg | 0.965876 | 0.906454 | 0.826976 | 0.843483 | 0.947620 | 0.899438 | 0.777073 | 0.817796 | 142.627439 |
| sgd_logreg | 0.961115 | 0.899889 | 0.815215 | 0.709821 | 0.914982 | 0.867789 | 0.724071 | 0.441794 | 28.999584 |
