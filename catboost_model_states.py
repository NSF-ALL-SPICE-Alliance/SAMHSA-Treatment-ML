#%%
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from pathlib import Path
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import rbo
#%%
# Load data
freq_use_rf_data = pd.read_feather("data/teds_d_completed_clean_no_freq1_na.feather")
#%%

# Identify region column
state_col = "stfips"  
unique_states = freq_use_rf_data[state_col].dropna().unique()

fips_state_name_dict = {
 '1': 'Alabama',
 '2': 'Alaska',
 '4': 'Arizona',
 '5': 'Arkansas',
 '6': 'California',
 '8': 'Colorado',
 '9': 'Connecticut',
 '10': 'Delaware',
 '12': 'Florida',
 '13': 'Georgia',
 '15': 'Hawaii',
 '16': 'Idaho',
 '17': 'Illinois',
 '18': 'Indiana',
 '19': 'Iowa',
 '20': 'Kansas',
 '21': 'Kentucky',
 '22': 'Louisiana',
 '23': 'Maine',
 '24': 'Maryland',
 '25': 'Massachusetts',
 '26': 'Michigan',
 '27': 'Minnesota',
 '28': 'Mississippi',
 '29': 'Missouri',
 '30': 'Montana',
 '31': 'Nebraska',
 '32': 'Nevada',
 '33': 'New Hampshire',
 '34': 'New Jersey',
 '35': 'New Mexico',
 '36': 'New York',
 '37': 'North Carolina',
 '38': 'North Dakota',
 '39': 'Ohio',
 '40': 'Oklahoma',
 '41': 'Oregon',
 '42': 'Pennsylvania',
 '44': 'Rhode Island',
 '45': 'South Carolina',
 '46': 'South Dakota',
 '47': 'Tennessee',
 '48': 'Texas',
 '49': 'Utah',
 '50': 'Vermont',
 '51': 'Virginia',
 '53': 'Washington',
 '54': 'West Virginia',
 '55': 'Wisconsin',
 '56': 'Wyoming',
 '60': 'American Samoa',
 '66': 'Guam',
 '69': 'Northern Mariana Islands',
 '72': 'Puerto Rico',
 '78': 'Virgin Islands'}


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
# Prep global model

# Set seed 
np.random.seed(123)

# Split into train and test
train_data, test_data = train_test_split(freq_use_rf_data,
                                         test_size=0.2,
                                         random_state=123,
                                         stratify=freq_use_rf_data["stfips"])

X_train = train_data.drop(columns=['freq1_d'])
y_train = train_data['freq1_d']

X_test = test_data.drop(columns=['freq1_d'])

y_test = test_data['freq1_d']

cat_features = X_train.select_dtypes(include='category').columns.tolist()

for col in cat_features:
    X_train[col] = X_train[col].astype(str).fillna("NaN")
    X_test[col] = X_test[col].astype(str).fillna("NaN")

train_pool = Pool(data=X_train, label=y_train, cat_features=cat_features)
test_pool = Pool(data=X_test, label=y_test, cat_features=cat_features)
#%% 
# Run / Load Global Model

# Define model parameters
params = {
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'iterations': 500,
    'depth': 6,
    'learning_rate': 0.1,
    'verbose': 100
}

## Train the model
global_model = CatBoostClassifier(**params)
global_model.fit(train_pool)

## Save the model
global_model.save_model("catboost_model_stfips_strat.bin")

## Load model
# global_model = CatBoostClassifier()
# global_model.load_model("catboost_model_stfips_strat.bin")  
#%%
# feature importance for global model
shap_values = global_model.get_feature_importance(test_pool, type='ShapValues')
shap_values = shap_values[:, :-1]
shap_values = pd.DataFrame(shap_values, columns=train_data.drop(columns=['freq1_d']).columns)
mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
importance_df = pd.DataFrame({
    'Feature': shap_values.columns,
    'Importance': mean_abs_shap
})

# importance_df["Feature"] = importance_df["Feature"].replace(feature_name_mapping)
importance_df_sorted = importance_df.sort_values(by='Importance', ascending=False).head(10)
top_global_features = importance_df_sorted['Feature'].tolist()
top_global_features
#%%
# filtered_dict = {k: v for k, v in fips_state_name_dict.items() if k in unique_states}
# filtered_dict = {k: v for k, v in fips_state_name_dict.items() if k == '35'} # for testing
filtered_dict = {k: v for k, v in fips_state_name_dict.items() if k in ['35', '36', '37']}

#%%
# Loop through each state 
for state_code, state_name in filtered_dict.items():
    # print(state_code)
    print(f"\nTraining model for state: {state_code}")
    region_data = freq_use_rf_data[freq_use_rf_data['stfips'] == state_code]
    # region_data = region_data.drop(columns='stfips')

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

    loc_model = CatBoostClassifier(**params)

    ## grid search: for future experimentation
    # grid = {'learning_rate': [0.03, 0.1],
    #     'depth': [6, 8],
    #     'l2_leaf_reg': [1, 5, 9]}

    # grid_search_result = model.grid_search(grid,
    #                                     X=train_pool,
    #                                     # y=train_labels,
    #                                     plot=True)

    # model = CatBoostClassifier(grid_search_result['params']  )
    ## end grid search

    loc_model.fit(train_pool, eval_set=test_pool, verbose=100)

    # # Save model
    # model_path = f"models/catboost_model_state_{state_name}.bin"
    # Path("models").mkdir(exist_ok=True)
    # model.save_model(model_path)

    local_predictions = loc_model.predict(test_pool, prediction_type='Probability')
    local_y_true = test_pool.get_label()
    local_positive_probs = local_predictions[:, 1]  # Get probabilities for class 1
    local_auc = roc_auc_score(local_y_true, local_positive_probs)

    # global predictions
    global_predictions = global_model.predict(test_pool, prediction_type='Probability')
    global_y_true = test_pool.get_label()
    global_positive_probs = global_predictions[:, 1]  # Get probabilities for class 1
    global_auc = roc_auc_score(global_y_true, global_positive_probs)

    print(f"\nGlobal AUC: {global_auc}")
    print(f"\nLocal AUC: {local_auc}")

    # feature importance for global model
    shap_values = loc_model.get_feature_importance(test_pool, type='ShapValues')
    shap_values = shap_values[:, :-1]
    shap_values = pd.DataFrame(shap_values, columns=train_data.drop(columns=['freq1_d']).columns)
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    importance_df = pd.DataFrame({
        'Feature': shap_values.columns,
        'Importance': mean_abs_shap
    })

    # importance_df["Feature"] = importance_df["Feature"].replace(feature_name_mapping)
    importance_df_sorted = importance_df.sort_values(by='Importance', ascending=False).head(10)
    top_local_features = importance_df_sorted['Feature'].tolist()

    plt.figure(figsize=(8, 12))
    sns.barplot(
        data=importance_df_sorted,
        x='Importance',
        y='Feature',
        palette='viridis'
    )
    plt.title(f'Feature Importance - {state_name}')
    plt.xlabel('Mean Absolute SHAP Value')
    plt.ylabel('Feature')
    plt.xticks(rotation=0)
    plt.yticks(fontsize=6)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'feature_importance_{state_name}.png')

    # rank biased overlap to compare top features against global model
    feature_rbo = rbo.RankingSimilarity(top_local_features, top_global_features).rbo()

    # save results to csv
    result_row = pd.DataFrame([{
    'stfips': state_code,
    'global_auc': global_auc,
    'local_auc': local_auc,
    'rbo': feature_rbo
    }])

    results_path = "model_performance_by_state.csv"

    if not os.path.exists(results_path):
        result_row.to_csv(results_path, index=False)
    else:
        result_row.to_csv(results_path, mode='a', index=False, header=False)
#%%
    ### raw var names
    # shap_values = model.get_feature_importance(test_pool, type='ShapValues')
    # # Drop the last column (it's the predicted value)
    # shap_values = shap_values[:, :-1]
    # X_test_renamed = X_test.rename(columns = feature_name_mapping)
    # X_test_np = X_test.values if hasattr(X_test, "values") else X_test
    # # Use shap.Explanation for proper formatting
    # explainer = shap.Explanation(values=shap_values, data=X_test_np, feature_names=X_test.columns)


    # # Generate the beeswarm plot and suppress automatic display
    # ax = shap.plots.beeswarm(explainer, show=False, max_display = 20)
    # plt.title(f"{state_name}")
    # plt.tight_layout()
    # plt.savefig(f"shap_{state_name}.png", dpi=300)
    # plt.close()
#%%
    # Assign column names from train_data (excluding 'freq1_d')
    # shap_values = pd.DataFrame(shap_values, columns=train_data.drop(columns=['freq1_d']).columns)

    # mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    # importance_df = pd.DataFrame({
    # 'Feature': shap_values.columns,
    # 'Importance': mean_abs_shap
    # })


    # importance_df["Feature"] = importance_df["Feature"].replace(feature_name_mapping)
    # importance_df_sorted = importance_df.sort_values(by='Importance', ascending=False)
    # plt.figure(figsize=(8, 12))
    # sns.barplot(
    #     data=importance_df_sorted,
    #     x='Importance',
    #     y='Feature',
    #     palette='viridis'
    # )
    # plt.title(f'Feature Importance - {state_name}')
    # plt.xlabel('Mean Absolute SHAP Value')
    # plt.ylabel('Feature')
    # plt.xticks(rotation=0)
    # plt.yticks(fontsize=6)
    # plt.tight_layout()
    # # plt.show()
    # plt.savefig(f'feature_importance_{state_name}.png')
    # plt.close()

#%%
