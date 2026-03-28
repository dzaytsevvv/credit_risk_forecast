# Model comparison (valid/test)

- Selection metric: **valid_pr_auc**
- Best model: **lgbm** (score=0.926048)

| model | valid_roc_auc | valid_pr_auc | valid_ks | valid_f1_at_0_5 | test_roc_auc | test_pr_auc | test_ks | test_f1_at_0_5 | fit_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lgbm | 0.974390 | 0.926048 | 0.849214 | 0.852309 | 0.975956 | 0.931790 | 0.847565 | 0.860500 | 10.219307 |
| logreg | 0.973037 | 0.914963 | 0.846868 | 0.825611 | 0.974393 | 0.921177 | 0.847275 | 0.829214 | 5.029602 |
| sgd_logreg | 0.971879 | 0.906859 | 0.845870 | 0.782383 | 0.973718 | 0.911986 | 0.844698 | 0.789650 | 1.176596 |
