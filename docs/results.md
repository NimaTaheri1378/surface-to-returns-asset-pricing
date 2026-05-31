# Results

The current package is a complete empirical handoff: full-sample data construction, surface fitting, GPU/SDF models, portfolio diagnostics, interpretation, and figures have all been generated and validated.

## Summary Metrics

| Area | Metric |
| --- | --- |
| Feature panel | 670,730 asset-month rows, 7,569 unique PERMNOs, 1996-01 to 2024-12 |
| External-enriched panel | 670,562 rows after FRED, BLS, BEA, EIA, and SEC enrichment |
| SSVI surface gate | 34,800 of 34,800 date-security surfaces passed |
| GPU return model | 467,085 OOS observations, 89 features, 19 folds, mean rank IC 0.0022 |
| Conditional autoencoder SDF | 467,085 OOS assets, 227 OOS months, pricing-error RMS 0.1158 |
| Proposal portfolio | 227 OOS months; TAQ-cost-adjusted long-short diagnostics reported alongside factor-pricing evidence |
| Asset-pricing inference | FF3, FF5+UMD, GRS, HJ distance, Newey-West, and block bootstrap diagnostics generated |
| Interpretation | Integrated gradients and TreeSHAP generated; top SHAP drivers include VIX changes, NFCI, rates, factors, and oil returns |
| Unit tests | 72 passing tests |

## Curated Tables

- [Surface monthly summary](assets/tables/surfaces/ssvi_monthly_summary.csv)
- [Proposal portfolio summary](assets/tables/backtests/proposal_buffered_summary.csv)
- [Proposal portfolio returns](assets/tables/backtests/proposal_buffered_portfolio.csv)
- [GPU decile summary](assets/tables/backtests/gpu_neural_decile_summary.csv)
- [FF5 proposal alphas](assets/tables/inference/ff5_proposal_alphas.csv)
- [Conditional autoencoder SDF monthly diagnostics](assets/tables/sdf/conditional_autoencoder_sdf_monthly.csv)
- [Conditional autoencoder SDF integrated gradients](assets/tables/sdf/conditional_autoencoder_sdf_integrated_gradients.csv)
- [TreeSHAP feature importance](assets/tables/interpretation/tree_shap_feature_importance.csv)
- [External API coverage](assets/tables/external/monthly_external_api_coverage.csv)
- [Reg SHO coefficients](assets/tables/regsho/regsho_pilot_did_coefficients.csv)

## Curated Reports

- [Static visual report](assets/reports/visual_report.html)
- [HTML slide deck](assets/reports/visual_slide_deck.html)
- [PDF figure package](assets/reports/visual_figure_package.pdf)
- [Visual artifact index](assets/reports/visual_artifact_index.html)
