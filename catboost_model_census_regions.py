#%%
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from pathlib import Path
import shap
import matplotlib.pyplot as plt
import seaborn as sns
#%%
# Load data
freq_use_rf_data = pd.read_feather("data/teds_d_completed_clean_no_freq1_na.feather")

# Set seed 
np.random.seed(123)

# Identify region column
region_col = "region"  
unique_regions = freq_use_rf_data[region_col].dropna().unique()

region_labels = {
    "0": "U.S. territories",
    "1": "Northeast",
    "2": "Midwest",
    "3": "South",
    "4": "West"
}

# Mapping dictionary for renaming features
feature_name_mapping = {
        "year": "Year",
        "caseid": "Case ID",
        "stfips": "State",
        "age": "Age",
        "services": "Service Type",
        "sub1": "Primary Substance at Admission",
        "sub2": "Secondary Substance at Admission",
        "sub3": "Tertiary Substance at Admission",
        "noprior": "No Prior Treatment Episodes",
        "psource": "Referral Source",
        "arrests": "Arrests in Past 30 Days Prior to Admission",
        "race": "Race",
        "ethnic": "Ethnicity",
        "educ": "Education Level",
        "employ": "Employment Status at Admission",
        "methuse": "Methamphetamine Use",
        "psyprob": "Psychiatric Problem in Past 30 Days",
        "vet": "Veteran Status",
        "livarag": "Living Arrangement at Admission",
        "priminc": "Primary Income Source",
        "hlthins": "Health Insurance",
        "marstat": "Marital Status",
        "daywait": "Number of Days Waiting for Admission",
        "route1": "Route of Administration for Primary Substance",
        "freq1": "Frequency of Use for Primary Substance at Admission",
        "frstuse1": "Age at First Use of Primary Substance",
        "route2": "Route of Administration for Secondary Substance",
        "freq2": "Frequency of Use for Secondary Substance at Admission",
        "frstuse2": "Age at First Use of Secondary Substance",
        "freq_atnd_self_help": "Attendance at Substance Use Self-Help Groups Prior to Admission",
        "dsmcrit": "Number of DSM-IV Criteria Met for Substance Use Disorder",
        "services_d": "Service Type at Discharge",
        "reason": "Reason for Discharge",
        "sub1_d": "Primary Substance at Discharge",
        "sub2_d": "Secondary Substance at Discharge",
        "sub3_d": "Tertiary Substance at Discharge",
        "employ_d": "Employment Status at Discharge",
        "livarag_d": "Living Arrangement at Discharge",
        "freq_atnd_self_help_d": "Attendance at Substance Use Self-Help Groups Prior to Discharge",
        "los_binned": "Binned Length of Stay",
        "arrests_d": "Arrests in Past 30 Days Prior to Discharge",
        "alcflg": "Alcohol Use Flag",
        "cokeflg": "Cocaine Use Flag",
        "marflg": "Marijuana Use Flag",
        "herflg": "Heroin Use Flag",
        "methflg": "Methamphetamine Use Flag",
        "opsynflg": "Other Opiate/Synthetic Use Flag",
        "pcpflg": "PCP Use Flag",
        "hallflg": "Hallucinogen Use Flag",
        "mthamflg": "Methadone Use Flag",
        "amphflg": "Amphetamine Use Flag",
        "stimflg": "Stimulant Use Flag",
        "benzflg": "Benzodiazepine Use Flag",
        "trnqflg": "Tranquilizer Use Flag",
        "barbflg": "Barbiturate Use Flag",
        "sedhpflg": "Sedative/Hypnotic Use Flag",
        "inhflg": "Inhalant Use Flag",
        "otcflg": "Over-the-Counter Drug Use Flag",
        "otherflg": "Other Drug Use Flag",
        "idu": "Injection Drug Use",
        "division": "Census Division",
        "region": "Census Region",
        "alcdrug": "Alcohol or Drug Use Flag",
        "gender": "Gender"
    }
#%%
# Loop through each region
for region_code, region_name in region_labels.items():
    print(region_code)
    print(f"\nTraining model for region: {region_name}")
    region_data = freq_use_rf_data[freq_use_rf_data['region'] == region_code]
    region_data = region_data.drop(columns='region')


    # Train-test split
    train_data, test_data = train_test_split(region_data, test_size=0.2, random_state=123)

    X_train = train_data.drop(columns=['freq1_d'])
    y_train = train_data['freq1_d']
    X_test = test_data.drop(columns=['freq1_d'])
    y_test = test_data['freq1_d']

    cat_features = X_train.select_dtypes(include='category').columns.tolist()

    # Handle categorical variables (convert to string and fill NA)
    for col in cat_features:
        X_train[col] = X_train[col].astype(str).fillna("NaN")
        X_test[col] = X_test[col].astype(str).fillna("NaN")

    train_pool = Pool(data=X_train, label=y_train, cat_features=cat_features)
    test_pool = Pool(data=X_test, label=y_test, cat_features=cat_features)

    params = {
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'iterations': 500,
        'depth': 6,
        'learning_rate': 0.1,
        'verbose': False
    }

    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=test_pool, verbose=100)

    # Save model
    model_path = f"models/catboost_model_region_{region_name}.bin"
    Path("models").mkdir(exist_ok=True)
    model.save_model(model_path)

    print(f"Saved model to {model_path}")

    shap_values = model.get_feature_importance(
    test_pool,
    type="ShapValues"
    )

    # Drop the last column from the NumPy array
    shap_values = shap_values[:, :-1]

    # Assign column names from train_data (excluding 'freq1_d')
    shap_values = pd.DataFrame(shap_values, columns=train_data.drop(columns=['freq1_d']).columns)

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    importance_df = pd.DataFrame({
    'Feature': shap_values.columns,
    'Importance': mean_abs_shap
    })


    importance_df["Feature"] = importance_df["Feature"].replace(feature_name_mapping)
    importance_df_sorted = importance_df.sort_values(by='Importance', ascending=False)
    plt.figure(figsize=(8, 12))
    sns.barplot(
        data=importance_df_sorted,
        x='Importance',
        y='Feature',
        palette='viridis'
    )
    plt.title(f'Feature Importance - {region_name}')
    plt.xlabel('Mean Absolute SHAP Value')
    plt.ylabel('Feature')
    plt.xticks(rotation=0)
    plt.yticks(fontsize=6)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'feature_importance_{region_name}.png')
#%%

importance_df_sorted = importance_df.sort_values(by='Importance', ascending=False)

#%%