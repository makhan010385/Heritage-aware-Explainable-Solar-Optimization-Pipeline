
# Results and Discussion

## Dataset characteristics
The study used 1,000 records describing historic/village-building characteristics, environmental renewable-energy potential, energy demand and retrofit-related variables. The continuous target, Optimal Solar Utilization, had a mean of 15.75% and a standard deviation of 5.54%, with an observed range of 1.79%–28.17%. The dataset contained 0 missing values and 0 duplicate rows.

## Predictive performance
For the deployment-safe feature set, the best model was **Ridge**, achieving R²=0.6785, RMSE=3.0235, MAE=2.5073, and MAPE=19.60%. The best full-feature benchmark achieved R²=0.7080. The comparison demonstrates why downstream performance indicators should be audited before being used as predictors.

## Explainability
The leading model-agnostic predictive drivers were: Installation_Area_m2, Solar_Potential_kWh_m2, Material, Building_Type, Year_Built. The importance analysis indicates which variables the model relies on most strongly, but these associations should not be interpreted as causal effects without controlled intervention data.

## Heritage-aware optimization
HESOP introduces a constrained counterfactual layer in which heritage-defining variables are held constant and installation area is varied computationally. This creates a minimal-intervention pathway for estimating how solar utilization could respond to retrofit intensity while avoiding arbitrary changes to the building's historic identity.

## Multi-objective decision support
The Pareto analysis retained 27 non-dominated records when simultaneously considering solar utilization, carbon reduction and payback period. This illustrates that sustainable retrofit planning is inherently multi-objective: a solution with high solar utilization is not necessarily optimal on every economic and environmental dimension.

## Research significance
The proposed framework moves from a conventional 'predict the target' workflow toward an explainable decision-support architecture combining prediction, feature attribution, constrained counterfactuals and Pareto screening. The approach is suitable for further validation on larger, real-world heritage-building datasets and can be extended with structural constraints, roof visibility, heritage regulations, local electricity tariffs and life-cycle carbon data.

## Limitations
The dataset has only 1,000 observations and does not explicitly encode heritage significance, roof geometry, structural load limits, conservation restrictions, shading/sky-view factor, electricity tariffs or installation costs. Therefore, the optimization outputs should be interpreted as computational screening results rather than engineering prescriptions. External validation and field measurements are required before practical deployment.
