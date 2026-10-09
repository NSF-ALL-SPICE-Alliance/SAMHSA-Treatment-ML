## Welcome

This is the repository for our publication [*Utilizing categorical boosting and SHAP to understand key predictors of frequency of use at discharge for completed substance abuse treatments in TEDS-D*](https://iopscience.iop.org/article/10.1088/3049-477X/ae9bc7)

## Data

The data is not pushed to this repository but is publicly available through SAMHSA [here](https://www.samhsa.gov/data/data-we-collect/teds-treatment-episode-data-set/datafiles?data_collection=1022). The steps from raw csv download to creating the finalized feather file can be found in the read_join_feather.R script

## Scripts

The [catboost.R](https://github.com/NSF-ALL-SPICE-Alliance/SAMHSA-Treatment-ML/blob/main/catboost.R) script will fully reproduce the primary model from the paper along with SHAP visualizations. 

Other figures may be reproduced with the circos_plot_generation.Rmd, heatmap_state_service_los_freq_use.Rmd, and catboost_model_states.py

## Questions

Please reach out with any questions to `connorflynn.chaminade.edu`




