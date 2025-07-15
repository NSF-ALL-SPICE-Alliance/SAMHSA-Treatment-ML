#%%

# !pip install catboost
#%%
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from pathlib import Path


# Load the Feather file previously generated from line 142 in Connor's script
freq_use_rf_data = pd.read_feather("data/teds_d_completed_clean_no_freq1_na.feather")
#%%
# Set seed for reproducibility
np.random.seed(123)

# Split into train and test
train_data, test_data = train_test_split(freq_use_rf_data, test_size=0.2, random_state=123)

# Separate features and labels
X_train = train_data.drop(columns='freq1_d')
y_train = train_data['freq1_d']

X_test = test_data.drop(columns='freq1_d')
y_test = test_data['freq1_d']

cat_features = X_train.select_dtypes(include='category').columns.tolist()
#%%

## ask Connor
for col in cat_features:
    X_train[col] = X_train[col].astype(str).fillna("NaN")
    X_test[col] = X_test[col].astype(str).fillna("NaN")
#%%
# Create CatBoost Pools
train_pool = Pool(data=X_train, label=y_train, cat_features=cat_features)
test_pool = Pool(data=X_test, label=y_test, cat_features=cat_features)
#%%
# Define model parameters
params = {
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'iterations': 500,
    'depth': 6,
    'learning_rate': 0.1,
    'verbose': 100
}
#%%
# # Train the model
# model = CatBoostClassifier(**params)
# model.fit(train_pool)
# # Save the model
# model.save_model("catboost_model.bin")
#%%
# load model
model = CatBoostClassifier()
model.load_model("catboost_model.bin")  

# Predict probabilities
predictions = model.predict(test_pool, prediction_type='Probability')
predicted_classes = np.where(predictions > 0.5, 1, 0)

#%%

