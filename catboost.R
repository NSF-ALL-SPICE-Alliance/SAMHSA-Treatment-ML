# Libraries ----
library(catboost)
library(tidyverse)
library(arrow)
library(here)
library(naniar)
library(shapviz)


### Read feather 
teds_d <- arrow::read_feather(here("data/teds_d_15_19.feather"))

### Check
glimpse(teds_d)

# Missing Data ----

missing_summary <- miss_var_summary(teds_d)


## Filter for completed treatments only

table(teds_d$reason)

teds_d_completed <- teds_d %>% 
  filter(reason == 1)


missing_summary_completed <- miss_var_summary(teds_d_completed)


## Drop columns where more than 55% of data is missing
columns_to_drop <- missing_summary_completed %>%
  filter(pct_miss > 55) %>%
  pull(variable)  # Extract the column names

columns_to_drop

# Drop columns with more than 55% missing data
teds_d_completed_clean <- teds_d_completed %>%
  select(-all_of(columns_to_drop))

# Drop disyr, caseid, and reason
teds_d_completed_clean <- teds_d_completed_clean %>% 
  select(-disyr, -caseid, -reason)


# Print the updated dataset
glimpse(teds_d_completed_clean)


### Bin Length of Stay

# # Custom binning for los
teds_d_completed_clean <- teds_d_completed_clean %>%
  mutate(
    los_binned = case_when(
      los == 1 ~ "1 day",                 # 1 day treatments
      los >= 2 & los <= 10 ~ "2-10 days", # 2 to 10 days
      los >= 11 & los <= 21 ~ "11-21 days", # 11 to 21 days
      los >= 22 & los <= 30 ~ "22-30 days", # 22 to 30 days
      los == 31 ~ "31-45 days",           # Existing codebook bins
      los == 32 ~ "46-60 days",
      los == 33 ~ "61-90 days",
      los == 34 ~ "91-120 days",
      los == 35 ~ "121-180 days",
      los == 36 ~ "181-365 days",
      los == 37 ~ ">365 days",
      TRUE ~ NA_character_                # Handle unexpected values
    ),
    los_binned = factor(
      los_binned,
      levels = c("1 day", "2-10 days", "11-21 days", "22-30 days",
                 "31-45 days", "46-60 days", "61-90 days",
                 "91-120 days", "121-180 days", "181-365 days", ">365 days"),
      ordered = FALSE
    )
  )

#
# # Verify the new bins
table(teds_d_completed_clean$los_binned)
#
#
# # Check the frequency table of the new variable
teds_d_completed_clean %>%
  select(los, los_binned) %>% 
  print(n = 50)
#
# ### Ensure factor
class(teds_d_completed_clean$los_binned)
#
# ### Drop los
#
teds_d_completed_clean <- teds_d_completed_clean %>%
  select(-los)

### Check freq1_d

table(teds_d_completed_clean$freq1_d)

sum(is.na(teds_d_completed_clean$freq1_d))

teds_d_completed_clean_no_freq1_na <- teds_d_completed_clean %>%
  filter(!is.na(freq1_d))

teds_d_completed_clean_no_freq1_na <- teds_d_completed_clean_no_freq1_na %>%
  mutate(
    freq1_d = case_when(
      freq1_d == 1 ~ "no use",
      freq1_d %in% c(2, 3) ~ "some/daily use",
      TRUE ~ as.character(freq1_d)  # Preserve other values as they are
    )
  )

table(teds_d_completed_clean_no_freq1_na$freq1_d)

### Mutate all to factor

teds_d_completed_clean_no_freq1_na <- teds_d_completed_clean_no_freq1_na %>%
  mutate(across(everything(), as.factor)) 

table(teds_d_completed_clean_no_freq1_na$freq1_d)

glimpse(teds_d_completed_clean_no_freq1_na)


### Prep freq1_d

# Convert target variable to numeric (starting from 0)
teds_d_completed_clean_no_freq1_na$freq1_d <- as.integer(as.factor(teds_d_completed_clean_no_freq1_na$freq1_d)) - 1

### Check
table(teds_d_completed_clean_no_freq1_na$freq1_d)

# Model ----

freq_use_rf_data <- teds_d_completed_clean_no_freq1_na

glimpse(freq_use_rf_data)

# Split Data into Training and Testing Sets

set.seed(123) # For reproducibility
train_indices <- sample(1:nrow(freq_use_rf_data), 0.8 * nrow(freq_use_rf_data))
train_data <- freq_use_rf_data[train_indices, ]
test_data <- freq_use_rf_data[-train_indices, ]



# Create CatBoost pools
train_pool <- catboost.load_pool(
  data = train_data %>% select(-freq1_d),
  label = train_data$freq1_d
)

test_pool <- catboost.load_pool(
  data = test_data %>% select(-freq1_d),
  label = test_data$freq1_d
)

## Train the CatBoost Model ----

# Define parameters
params <- list(
  loss_function = "Logloss",       # Binary classification
  eval_metric = "AUC",            # Metric for evaluation
  iterations = 500,               # Number of boosting iterations
  depth = 6,                      # Depth of trees
  learning_rate = 0.1,            # Learning rate
  verbose = 100                   # Log every 100 iterations
)

# # # Train the model
# model <- catboost.train(
#    learn_pool = train_pool,
#    params = params
#  )
#
# # # Save the model
# # # Define the file path for saving the model
model_path <- here("models", "catboost_model.bin")
# # #
# # #
# # # # Save the model
#  catboost.save_model(
#   model = model,
#   model_path = model_path
# )

# Load Model
model <- catboost.load_model(model_path)

## Evaluate the Model ----

# Predict probabilities for the test set
predictions <- catboost.predict(model, test_pool, prediction_type = "Probability")

# Convert probabilities to binary predictions (threshold 0.5)
predicted_classes <- ifelse(predictions > 0.5, 1, 0)

# Compute confusion matrix
confusion_matrix <- table(Predicted = predicted_classes, Actual = test_data$freq1_d)
print(confusion_matrix)

# Compute AUC
library(pROC)
auc <- roc(test_data$freq1_d, predictions)
print(paste("AUC:", auc$auc))


## Feature Importance ----



# Get feature importance
shap_values <- catboost.get_feature_importance(
  model,
  pool = test_pool,
  type = "ShapValues"
)

# # Removing baseline
shap_values <- shap_values[, -ncol(shap_values)]


colnames(shap_values) <- colnames(train_data%>% select(-freq1_d))


# 
# # Identify the column index of freq1_d in train_data
# freq1_d_index <- which(colnames(test_data) == "freq1_d")
# 
# 
# # Remove the freq1_d column from SHAP values (adjust for baseline removal, if applicable)
# shap_values <- shap_values[, -freq1_d_index]

ncol(shap_values)

# Aggregate SHAP values: Mean absolute SHAP value for each feature
mean_abs_shap <- colMeans(abs(shap_values))

# Create a data frame for visualization
importance_df <- data.frame(
  Feature = colnames(shap_values),  # Features from test data
  Importance = mean_abs_shap
)

# Recode for plot

importance_df <- importance_df %>%
  mutate(Feature = case_when(
    Feature == "year" ~ "Year",
    Feature == "caseid" ~ "Case ID",
    Feature == "stfips" ~ "State",
    Feature == "age" ~ "Age",
    Feature == "services" ~ "Service Type",
    Feature == "sub1" ~ "Primary Substance at Admission",
    Feature == "sub2" ~ "Secondary Substance at Admission",
    Feature == "sub3" ~ "Tertiary Substance at Admission",
    Feature == "noprior" ~ "No Prior Treatment Episodes",
    Feature == "psource" ~ "Referral Source",
    Feature == "arrests" ~ "Arrests in Past 30 Days Prior to Admission",
    Feature == "race" ~ "Race",
    Feature == "ethnic" ~ "Ethnicity",
    Feature == "educ" ~ "Education Level",
    Feature == "employ" ~ "Employment Status at Admission",
    Feature == "methuse" ~ "Methamphetamine Use",
    Feature == "psyprob" ~ "Psychiatric Problem in Past 30 Days",
    Feature == "vet" ~ "Veteran Status",
    Feature == "livarag" ~ "Living Arrangement at Admission",
    Feature == "priminc" ~ "Primary Income Source",
    Feature == "hlthins" ~ "Health Insurance",
    Feature == "marstat" ~ "Marital Status",
    Feature == "daywait" ~ "Number of Days Waiting for Admission",
    Feature == "route1" ~ "Route of Administration for Primary Substance",
    Feature == "freq1" ~ "Frequency of Use for Primary Substance at Admission",
    Feature == "frstuse1" ~ "Age at First Use of Primary Substance",
    Feature == "route2" ~ "Route of Administration for Secondary Substance",
    Feature == "freq2" ~ "Frequency of Use for Secondary Substance at Admission",
    Feature == "frstuse2" ~ "Age at First Use of Secondary Substance",
    Feature == "freq_atnd_self_help" ~ "Attendance at Substance Use Self-Help Groups Prior to Admission",
    Feature == "dsmcrit" ~ "Number of DSM-IV Criteria Met for Substance Use Disorder",
    Feature == "services_d" ~ "Service Type at Discharge",
    Feature == "reason" ~ "Reason for Discharge",
    Feature == "sub1_d" ~ "Primary Substance at Discharge",
    Feature == "sub2_d" ~ "Secondary Substance at Discharge",
    Feature == "sub3_d" ~ "Tertiary Substance at Discharge",
    Feature == "employ_d" ~ "Employment Status at Discharge",
    Feature == "livarag_d" ~ "Living Arrangement at Discharge",
    Feature == "freq_atnd_self_help_d" ~ "Attendance at Substance Use Self-Help Groups Prior to Discharge",
    Feature == "los_binned" ~ "Binned Length of Stay",
    Feature == "arrests_d" ~ "Arrests in Past 30 Days Prior to Discharge",
    Feature == "alcflg" ~ "Alcohol Use Flag",
    Feature == "cokeflg" ~ "Cocaine Use Flag",
    Feature == "marflg" ~ "Marijuana Use Flag",
    Feature == "herflg" ~ "Heroin Use Flag",
    Feature == "methflg" ~ "Methamphetamine Use Flag",
    Feature == "opsynflg" ~ "Other Opiate/Synthetic Use Flag",
    Feature == "pcpflg" ~ "PCP Use Flag",
    Feature == "hallflg" ~ "Hallucinogen Use Flag",
    Feature == "mthamflg" ~ "Methadone Use Flag",
    Feature == "amphflg" ~ "Amphetamine Use Flag",
    Feature == "stimflg" ~ "Stimulant Use Flag",
    Feature == "benzflg" ~ "Benzodiazepine Use Flag",
    Feature == "trnqflg" ~ "Tranquilizer Use Flag",
    Feature == "barbflg" ~ "Barbiturate Use Flag",
    Feature == "sedhpflg" ~ "Sedative/Hypnotic Use Flag",
    Feature == "inhflg" ~ "Inhalant Use Flag",
    Feature == "otcflg" ~ "Over-the-Counter Drug Use Flag",
    Feature == "otherflg" ~ "Other Drug Use Flag",
    Feature == "idu" ~ "Injection Drug Use",
    Feature == "division" ~ "Census Division",
    Feature == "region" ~ "Census Region",
    Feature == "alcdrug" ~ "Alcohol or Drug Use Flag",
    Feature == "gender" ~ "Gender",
    TRUE ~ Feature  # Keep unmatched features as is
  ))


# Plot feature importance
feature_importance_plot <- ggplot(importance_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Feature Importance Based on Mean Absolute SHAP Values", x = "Feature", y = "Mean Absolute SHAP Value") +
  theme_bw() +
  theme(axis.text.y = element_text(size =6))
feature_importance_plot

## Save plot

# ggsave(filename = here("plots", "feature_importance_plot.png"),
#        plot = feature_importance_plot,
#        dpi = 300,
#        width = 10,
#        height = 6)


## Partial dependence  ----
#Get SHAP values for the test data
# shap_values <- catboost.get_feature_importance(
#   model,
#   pool = test_pool,
#   type = "ShapValues"
# )

### STFIPS ----

# Extract SHAP values for a specific feature (e.g., "stfips")
shap_feature_stfips <- shap_values[, which(colnames(train_data) == "stfips")]

# Combine SHAP values with test data for visualization
shap_df_stfips <- data.frame(
  FeatureValue = test_data$stfips,
  ShapValue = shap_feature_stfips
)

shap_df_stfips <- shap_df_stfips %>%
  group_by(FeatureValue) %>%
  mutate(median = median(ShapValue)) %>%
  ungroup()

shap_df_stfips <- shap_df_stfips %>%
  mutate(FeatureValue = case_when(
    FeatureValue == "1" ~ "Alabama",
    FeatureValue == "2" ~ "Alaska",
    FeatureValue == "4" ~ "Arizona",
    FeatureValue == "5" ~ "Arkansas",
    FeatureValue == "6" ~ "California",
    FeatureValue == "8" ~ "Colorado",
    FeatureValue == "9" ~ "Connecticut",
    FeatureValue == "10" ~ "Delaware",
    FeatureValue == "11" ~ "District of Columbia",
    FeatureValue == "12" ~ "Florida",
    FeatureValue == "13" ~ "Georgia",
    FeatureValue == "15" ~ "Hawaii",
    FeatureValue == "16" ~ "Idaho",
    FeatureValue == "17" ~ "Illinois",
    FeatureValue == "18" ~ "Indiana",
    FeatureValue == "19" ~ "Iowa",
    FeatureValue == "20" ~ "Kansas",
    FeatureValue == "21" ~ "Kentucky",
    FeatureValue == "22" ~ "Louisiana",
    FeatureValue == "23" ~ "Maine",
    FeatureValue == "24" ~ "Maryland",
    FeatureValue == "25" ~ "Massachusetts",
    FeatureValue == "26" ~ "Michigan",
    FeatureValue == "27" ~ "Minnesota",
    FeatureValue == "28" ~ "Mississippi",
    FeatureValue == "29" ~ "Missouri",
    FeatureValue == "30" ~ "Montana",
    FeatureValue == "31" ~ "Nebraska",
    FeatureValue == "32" ~ "Nevada",
    FeatureValue == "33" ~ "New Hampshire",
    FeatureValue == "34" ~ "New Jersey",
    FeatureValue == "35" ~ "New Mexico",
    FeatureValue == "36" ~ "New York",
    FeatureValue == "37" ~ "North Carolina",
    FeatureValue == "38" ~ "North Dakota",
    FeatureValue == "39" ~ "Ohio",
    FeatureValue == "40" ~ "Oklahoma",
    FeatureValue == "42" ~ "Pennsylvania",
    FeatureValue == "44" ~ "Rhode Island",
    FeatureValue == "45" ~ "South Carolina",
    FeatureValue == "46" ~ "South Dakota",
    FeatureValue == "47" ~ "Tennessee",
    FeatureValue == "48" ~ "Texas",
    FeatureValue == "49" ~ "Utah",
    FeatureValue == "50" ~ "Vermont",
    FeatureValue == "51" ~ "Virginia",
    FeatureValue == "53" ~ "Washington",
    FeatureValue == "55" ~ "Wisconsin",
    FeatureValue == "56" ~ "Wyoming",
    FeatureValue == "72" ~ "Puerto Rico",
    TRUE ~ FeatureValue  # Keep as is for any unmatched values
  ))

# Plot partial dependence
#library(ggplot2)
partial_dependence_stfips<- ggplot(shap_df_stfips, aes(x = reorder(FeatureValue, median), y = ShapValue)) +
  geom_boxplot() +
  labs(title = "Partial Dependence for stfips",
       x = "Feature Value (stfips)",
       y = "SHAP Value (Impact on Prediction)") +
  coord_flip() +
  theme_bw()
partial_dependence_stfips

## Save plot

# ggsave(filename = here("plots", "partial_dependence_stfips.png"),
#        plot = partial_dependence_stfips,
#        dpi = 300,
#        width = 10,
#        height = 6)

### Freq1 ----

# Extract SHAP values for a specific feature (e.g., "freq1")
shap_feature_freq1 <- shap_values[, which(colnames(train_data) == "freq1")]

# Combine SHAP values with test data for visualization
shap_df_freq1 <- data.frame(
  FeatureValue = test_data$freq1,
  ShapValue = shap_feature_freq1
)

shap_df_freq1 <- shap_df_freq1 %>%
  group_by(FeatureValue) %>%
  mutate(median = median(ShapValue)) %>%
  ungroup()

shap_df_freq1 <- shap_df_freq1 %>%
  mutate(FeatureValue = case_when(
    FeatureValue == "1" ~ "No use in the past month",
    FeatureValue == "2" ~ "Some use",
    FeatureValue == "3" ~ "Daily use",
    TRUE ~ FeatureValue  # Keep as is for any unmatched values
  ))

# Plot partial dependence
#library(ggplot2)
partial_dependence_freq1 <- ggplot(shap_df_freq1, aes(x = reorder(FeatureValue, median), y = ShapValue)) +
  geom_boxplot() +
  labs(title = "Partial Dependence for frequency of use at admission",
       x = "Feature Value (freq1)",
       y = "SHAP Value (Impact on Prediction)") +
  coord_flip() +
  theme_bw()
partial_dependence_freq1

# ggsave(filename = here("plots", "partial_dependence_freq1.png"),
#        plot = partial_dependence_freq1,
#        dpi = 300,
#        width = 10,
#        height = 6)



### LOS ----
# Extract SHAP values for a specific feature (e.g., "los")
shap_feature_los_binned <- shap_values[, 61]

# Combine SHAP values with test data for visualization
shap_df_los_binned <- data.frame(
  FeatureValue = test_data$los_binned,
  ShapValue = shap_feature_los_binned
)

shap_df_los_binned <- shap_df_los_binned %>%
  group_by(FeatureValue) %>%
  mutate(median = median(ShapValue)) %>%
  ungroup()

partial_dependence_los <- ggplot(shap_df_los_binned, aes(x = reorder(FeatureValue, median), y = ShapValue)) +
  geom_boxplot() +
  labs(title = "Partial Dependence for length of stay",
       x = "Feature Value (los)",
       y = "SHAP Value (Impact on Prediction)") +
  coord_flip() +
  theme_bw()
partial_dependence_los

# ggsave(filename = here("plots", "partial_dependence_los.png"),
#        plot = partial_dependence_los,
#        dpi = 300,
#        width = 10,
#        height = 6)

### self help ----

# Extract SHAP values for a specific feature (e.g., "freq_atnd_self_help_d")
shap_feature_self_help <- shap_values[, which(colnames(train_data) == "freq_atnd_self_help_d")]

# Combine SHAP values with test data for visualization
shap_df_self_help <- data.frame(
  FeatureValue = test_data$freq_atnd_self_help_d,
  ShapValue = shap_feature_self_help
)

shap_df_self_help <- shap_df_self_help %>%
  group_by(FeatureValue) %>%
  mutate(median = median(ShapValue)) %>%
  ungroup()

shap_df_self_help <- shap_df_self_help %>%
  mutate(FeatureValue = case_when(
    FeatureValue == "1" ~ "No attendance",
    FeatureValue == "2" ~ "1-3 times in the past month",
    FeatureValue == "3" ~ "4-7 times in the past month",
    FeatureValue == "4" ~ "8-30 times in the past month",
    FeatureValue == "5" ~ "Some attendance, frequency is unknown",
    TRUE ~ FeatureValue  # Keep as is for any unmatched values
  ))

# Plot partial dependence
#library(ggplot2)
partial_dependence_self_help <- ggplot(shap_df_self_help, aes(x = reorder(FeatureValue, median), y = ShapValue)) +
  geom_boxplot() +
  labs(title = "Partial Dependence for frequency of self help at discharge",
       x = "Feature Value (self help)",
       y = "SHAP Value (Impact on Prediction)") +
  coord_flip() +
  theme_bw()
partial_dependence_self_help

# ggsave(filename = here("plots", "partial_dependence_self_help.png"),
#        plot = partial_dependence_self_help,
#        dpi = 300,
#        width = 10,
#        height = 6)


### services d ----

# Extract SHAP values for a specific feature (e.g., "freq_atnd_self_help_d")
shap_feature_services_d <- shap_values[, which(colnames(train_data) == "services_d")]

# Combine SHAP values with test data for visualization
shap_df_services_d <- data.frame(
  FeatureValue = test_data$services_d,
  ShapValue = shap_feature_services_d
)

shap_df_services_d <- shap_df_services_d %>%
  group_by(FeatureValue) %>%
  mutate(median = median(ShapValue)) %>%
  ungroup()

shap_df_services_d <- shap_df_services_d %>%
  mutate(FeatureValue = case_when(
    FeatureValue == "1" ~ "Detox, 24-hour, hospital inpatient",
    FeatureValue == "2" ~ "Detox, 24-hour, free-standing residential ",
    FeatureValue == "3" ~ "Rehab/residential, hospital (non-detox)",
    FeatureValue == "4" ~ "Rehab/residential, short term (30 days or fewer)",
    FeatureValue == "5" ~ "Rehab/residential, long term (more than 30 days) ",
    FeatureValue == "6" ~ "Ambulatory, intensive outpatient",
    FeatureValue == "7" ~ "Ambulatory, non-intensive outpatient",
    FeatureValue == "8" ~ "Ambulatory, detoxification",
    TRUE ~ FeatureValue  # Keep as is for any unmatched values
  ))

# Plot partial dependence
#library(ggplot2)
partial_dependence_services_d <- ggplot(shap_df_services_d, aes(x = reorder(FeatureValue, median), y = ShapValue)) +
  geom_boxplot() +
  labs(title = "Partial Dependence for frequency of service at discharge",
       x = "Feature Value (service)",
       y = "SHAP Value (Impact on Prediction)") +
  coord_flip() +
  theme_bw()
partial_dependence_services_d

# ggsave(filename = here("plots", "partial_dependence_services_d.png"),
#        plot = partial_dependence_services_d,
#        dpi = 300,
#        width = 10,
#        height = 6)

### livarag d ----

# Extract SHAP values for a specific feature (e.g., "livarag_d")
shap_feature_livarag_d <- shap_values[, which(colnames(train_data) == "livarag_d")]

# Combine SHAP values with test data for visualization
shap_df_livarag_d <- data.frame(
  FeatureValue = test_data$livarag_d,
  ShapValue = shap_feature_livarag_d
)

shap_df_livarag_d <- shap_df_livarag_d%>%
  group_by(FeatureValue) %>%
  mutate(median = median(ShapValue)) %>%
  ungroup()

shap_df_livarag_d <- shap_df_livarag_d %>%
  mutate(FeatureValue = case_when(
    FeatureValue == "1" ~ "Homeless",
    FeatureValue == "2" ~ "Dependent living",
    FeatureValue == "3" ~ "Independent living",
    TRUE ~ FeatureValue  # Keep as is for any unmatched values
  ))

# Plot partial dependence
#library(ggplot2)
partial_dependence_livarag_d <- ggplot(shap_df_livarag_d, aes(x = reorder(FeatureValue, median), y = ShapValue)) +
  geom_boxplot() +
  labs(title = "Partial Dependence for frequency of living arrangement at discharge",
       x = "Feature Value (livarag_d)",
       y = "SHAP Value (Impact on Prediction)") +
  coord_flip() +
  theme_bw()
partial_dependence_livarag_d

# ggsave(filename = here("plots", "partial_dependence_livarag_d.png"),
#        plot = partial_dependence_livarag_d,
#        dpi = 300,
#        width = 10,
#        height = 6)





### division ----


# Extract SHAP values for a specific feature (e.g., "division")
shap_feature_division <- shap_values[, which(colnames(train_data) == "division")]

# Combine SHAP values with test data for visualization
shap_df_division <- data.frame(
  FeatureValue = test_data$division,
  ShapValue = shap_feature_division
)

shap_df_division <- shap_df_division %>%
  group_by(FeatureValue) %>%
  mutate(median = median(ShapValue)) %>%
  ungroup()

shap_df_division <- shap_df_division%>%
  mutate(FeatureValue = case_when(
    FeatureValue == "0" ~ "U.S. territories",
    FeatureValue == "1" ~ "New England",
    FeatureValue == "2" ~ "Middle Atlantic",
    FeatureValue == "3" ~ "East North Central",
    FeatureValue == "4" ~ "West North Central",
    FeatureValue == "5" ~ "South Atlantic",
    FeatureValue == "6" ~ "East South Central",
    FeatureValue == "7" ~ "West South Central",
    FeatureValue == "8" ~ "Mountain",
    FeatureValue == "9" ~ "Pacific",
    TRUE ~ FeatureValue  # Keep as is for any unmatched values
  ))

# Plot partial dependence
#library(ggplot2)
partial_dependence_division <- ggplot(shap_df_division, aes(x = reorder(FeatureValue, median), y = ShapValue)) +
  geom_boxplot() +
  labs(title = "Partial Dependence for division",
       x = "Feature Value (division)",
       y = "SHAP Value (Impact on Prediction)") +
  coord_flip() +
  theme_bw()
partial_dependence_division

# ggsave(filename = here("plots", "partial_dependence_division.png"),
#        plot = partial_dependence_division,
#        dpi = 300,
#        width = 10,
#        height = 6)




### services ----

# Extract SHAP values for a specific feature (e.g., "services")
shap_feature_services <- shap_values[, which(colnames(train_data) == "services")]

# Combine SHAP values with test data for visualization
shap_df_services <- data.frame(
  FeatureValue = test_data$services,
  ShapValue = shap_feature_services
)

shap_df_services <- shap_df_services%>%
  group_by(FeatureValue) %>%
  mutate(median = median(ShapValue)) %>%
  ungroup()

shap_df_services <- shap_df_services %>%
  mutate(FeatureValue = case_when(
    FeatureValue == "1" ~ "Detox, 24-hour, hospital inpatient",
    FeatureValue == "2" ~ "Detox, 24-hour, free-standing residential ",
    FeatureValue == "3" ~ "Rehab/residential, hospital (non-detox)",
    FeatureValue == "4" ~ "Rehab/residential, short term (30 days or fewer)",
    FeatureValue == "5" ~ "Rehab/residential, long term (more than 30 days) ",
    FeatureValue == "6" ~ "Ambulatory, intensive outpatient",
    FeatureValue == "7" ~ "Ambulatory, non-intensive outpatient",
    FeatureValue == "8" ~ "Ambulatory, detoxification",
    TRUE ~ FeatureValue  # Keep as is for any unmatched values
  ))

# Plot partial dependence
#library(ggplot2)
partial_dependence_services <- ggplot(shap_df_services, aes(x = reorder(FeatureValue, median), y = ShapValue)) +
  geom_boxplot() +
  labs(title = "Partial Dependence for frequency of service at admission",
       x = "Feature Value (service)",
       y = "SHAP Value (Impact on Prediction)") +
  coord_flip() +
  theme_bw()

partial_dependence_services

# ggsave(filename = here("plots", "partial_dependence_services.png"),
#        plot = partial_dependence_services,
#        dpi = 300,
#        width = 10,
#        height = 6)


### psource ----

# Extract SHAP values for a specific feature (e.g., "livarag_d")
shap_feature_psource <- shap_values[, which(colnames(train_data) == "psource")]

# Combine SHAP values with test data for visualization
shap_df_psource <- data.frame(
  FeatureValue = test_data$psource,
  ShapValue = shap_feature_psource
)

shap_df_psource <- shap_df_psource %>%
  group_by(FeatureValue) %>%
  mutate(median = median(ShapValue)) %>%
  ungroup()

shap_df_psource <- shap_df_psource %>%
  mutate(FeatureValue = case_when(
    FeatureValue == "1" ~ "Individual (includes self-referral)",
    FeatureValue == "2" ~ "Alcohol/drug use care provider",
    FeatureValue == "3" ~ "Other health care provider",
    FeatureValue == "4" ~ "School (educational)",
    FeatureValue == "5" ~ "Employer/EAP",
    FeatureValue == "6" ~ "Other community referral",
    FeatureValue == "7" ~ "Court/criminal justice referral/DUI/DWI",
    TRUE ~ FeatureValue  # Keep as is for any unmatched values
  ))

# Plot partial dependence
#library(ggplot2)
partial_dependence_psource <- ggplot(shap_df_psource, aes(x = reorder(FeatureValue, median), y = ShapValue)) +
  geom_boxplot() +
  labs(title = "Partial Dependence for referral source",
       x = "Feature Value (psource)",
       y = "SHAP Value (Impact on Prediction)") +
  coord_flip() +
  theme_bw()
partial_dependence_psource

# ggsave(filename = here("plots", "partial_dependence_psource.png"),
#        plot = partial_dependence_psource,
#        dpi = 300,
#        width = 10,
#        height = 6)

### hlthins ----

# Extract SHAP values for a specific feature (e.g., "hlthins")
shap_feature_hlthins <- shap_values[, which(colnames(train_data) == "hlthins")]

# Combine SHAP values with test data for visualization
shap_df_hlthins <- data.frame(
  FeatureValue = test_data$hlthins,
  ShapValue = shap_feature_hlthins
)

shap_df_hlthins <- shap_df_hlthins %>%
  group_by(FeatureValue) %>%
  mutate(median = median(ShapValue)) %>%
  ungroup()

shap_df_hlthins <- shap_df_hlthins %>%
  mutate(FeatureValue = case_when(
    FeatureValue == "1" ~ "Private insurance, Blue Cross/Blue Shield, HMO",
    FeatureValue == "2" ~ "Medicaid",
    FeatureValue == "3" ~ "Medicare, other (e.g. TRICARE, CHAMPUS)",
    FeatureValue == "4" ~ "None",
    TRUE ~ FeatureValue  # Keep as is for any unmatched values
  ))

# Plot partial dependence
#library(ggplot2)
partial_dependence_hlthins <- ggplot(shap_df_hlthins, aes(x = reorder(FeatureValue, median), y = ShapValue)) +
  geom_boxplot() +
  labs(title = "Partial Dependence for health insurance",
       x = "Feature Value (health insurance)",
       y = "SHAP Value (Impact on Prediction)") +
  coord_flip() +
  theme_bw()
partial_dependence_hlthins

# ggsave(filename = here("plots", "partial_dependence_hlthins.png"),
#        plot = partial_dependence_hlthins,
#        dpi = 300,
#        width = 10,
#        height = 6)

### employment ----

# Extract SHAP values for a specific feature (e.g., "employ_d")
shap_feature_employ_d <- shap_values[, which(colnames(train_data) == "employ_d")]

# Combine SHAP values with test data for visualization
shap_df_employ_d <- data.frame(
  FeatureValue = test_data$employ_d,
  ShapValue = shap_feature_employ_d
)

shap_df_employ_d <- shap_df_employ_d %>%
  group_by(FeatureValue) %>%
  mutate(median = median(ShapValue)) %>%
  ungroup()

shap_df_employ_d <- shap_df_employ_d %>%
  mutate(FeatureValue = case_when(
    FeatureValue == "1" ~ "Full-time",
    FeatureValue == "2" ~ "Part-time",
    FeatureValue == "3" ~ "Unemployed",
    FeatureValue == "4" ~ "Not in Labor Force",
    TRUE ~ FeatureValue  # Keep as is for any unmatched values
  ))

# Plot partial dependence
#library(ggplot2)
partial_dependence_employ_d <- ggplot(shap_df_employ_d, aes(x = reorder(FeatureValue, median), y = ShapValue)) +
  geom_boxplot() +
  labs(title = "Partial Dependence for employment",
       x = "Feature Value (employment)",
       y = "SHAP Value (Impact on Prediction)") +
  coord_flip() +
  theme_bw()
partial_dependence_employ_d

# ggsave(filename = here("plots", "partial_dependence_employ_d.png"),
#        plot = partial_dependence_employ_d,
#        dpi = 300,
#        width = 10,
#        height = 6)


### primary income ----

# Extract SHAP values for a specific feature (e.g., "employ_d")
shap_feature_priminc<- shap_values[, which(colnames(train_data) == "priminc")]

# Combine SHAP values with test data for visualization
shap_df_priminc<- data.frame(
  FeatureValue = test_data$priminc,
  ShapValue = shap_feature_priminc
)

shap_df_priminc<- shap_df_priminc%>%
  group_by(FeatureValue) %>%
  mutate(median = median(ShapValue)) %>%
  ungroup()

shap_df_priminc<- shap_df_priminc %>%
  mutate(FeatureValue = case_when(
    FeatureValue == "1" ~ "Wages/salary",
    FeatureValue == "2" ~ "Public assistance",
    FeatureValue == "3" ~ "Retirement/pension, disability",
    FeatureValue == "4" ~ "Other",
    FeatureValue == "5" ~ "None",
    TRUE ~ FeatureValue  # Keep as is for any unmatched values
  ))

# Plot partial dependence
#library(ggplot2)
partial_dependence_priminc <- ggplot(shap_df_priminc, aes(x = reorder(FeatureValue, median), y = ShapValue)) +
  geom_boxplot() +
  labs(title = "Partial Dependence for primary income",
       x = "Feature Value (primary income)",
       y = "SHAP Value (Impact on Prediction)") +
  coord_flip() +
  theme_bw()
partial_dependence_priminc

# ggsave(filename = here("plots", "partial_dependence_priminc.png"),
#        plot = partial_dependence_priminc,
#        dpi = 300,
#        width = 10,
#        height = 6)

# faceted ---- 


# Combine all processed SHAP dataframes
shap_df_combined <- bind_rows(
  shap_df_self_help %>% mutate(Feature = "Self-Help Attendance"),
  shap_df_hlthins %>% mutate(Feature = "Health Insurance"),
  shap_df_livarag_d %>% mutate(Feature = "Living Arrangement"),
  shap_df_employ_d %>% mutate(Feature = "Employment Status"),
  shap_df_psource %>% mutate(Feature = "Referral Source")
)

# Create faceted Partial Dependence Plot
pdp_faceted <- ggplot(shap_df_combined, aes(x = reorder(FeatureValue, median), y = ShapValue)) +
  geom_boxplot() +
  facet_wrap(~ Feature, scales = "free_y") +
  labs(title = "Partial Dependence Plots for Key Predictors",
       x = "Feature Value",
       y = "SHAP Value (Impact on Prediction)") +
  coord_flip() +
  theme_bw()

# Display plot
pdp_faceted

# Save the plot
# ggsave(filename = here("plots", "partial_dependence_faceted.png"),
#        plot = pdp_faceted,
#        dpi = 300,
#        width = 12,
#        height = 8)






# shapviz ----

# shap_values <- catboost.get_feature_importance(
#   model,
#   pool = test_pool,
#   type = "ShapValues"
# )
# 
# # Remove last column (base value) from SHAP values
# shap_values <- shap_values[, -ncol(shap_values)]
# 
# # Ensure feature names are correctly assigned
# colnames(shap_values) <- colnames(train_data %>% select(-freq1_d))

shap_obj <- shapviz(shap_values, X = test_data %>% select(-freq1_d))

sv_importance(shap_obj) + ggtitle("SHAP Feature Importance (shapviz)")


sv_waterfall(shap_obj, row_id = 500) + ggtitle("SHAP Waterfall Plot with Renamed Features")
sv_waterfall(shap_obj, row_id = 2574) + ggtitle("SHAP Waterfall Plot with Renamed Features")

test_data_shap_lookup <- test_data %>% 
  rowid_to_column(var = "id")



sv_waterfall(shap_obj, row_id = 33736) + ggtitle("SHAP Waterfall Plot")
sv_waterfall(shap_obj, row_id = 128632) + ggtitle("SHAP Waterfall Plot")





shapsv_waterfall(shap_obj, row_id = 34995) + ggtitle("SHAP Waterfall Plot")
sv_waterfall(shap_obj, row_id = 1) + ggtitle("SHAP Waterfall Plot")
sv_waterfall(shap_obj, row_id = 2) + ggtitle("SHAP Waterfall Plot")


sv_dependence(shap_obj, v = "stfips") + ggtitle("SHAP Dependence for STFIPS")
#sv_dependence(shap_obj, v = "freq1_d") + ggtitle("SHAP Dependence for Frequency of Use")
sv_dependence(shap_obj, v = "services_d") + ggtitle("SHAP Dependence for Length of Stay")


sv_dependence(shap_obj, v = "stfips", color_var = "los_binned") +
  ggtitle("SHAP Dependence for Services at Discharge and LOS")

sv_dependence(shap_obj, v = "stfips", color_var = "services_d") 

sv_dependence(shap_obj, v = "services_d", color_var = "los_binned") 

x_subgroups <- split(shap_obj, f = test_data$freq1_d)
sv_importance(x_subgroups)

shap_interactions <- catboost.get_feature_importance(
  model,
  pool = test_pool,
  type = "Interaction"
)
feature_names <- colnames(test_data)

# Convert indices to actual feature names
shap_interactions_named <- data.frame(
  Feature1 = feature_names[shap_interactions[, "feature1_index"]],
  Feature2 = feature_names[shap_interactions[, "feature2_index"]],
  Score = shap_interactions[, "score"]
)

# geoshapley ----

# Save SHAP values and test data as Feather or CSV
# write_feather(as.data.frame(shap_values), here("data", "shap_values.feather"))
# write_feather(test_data %>% select(-freq1_d), here("data", "X_test.feather"))
# write_feather(test_data %>% select(stfips), here("data", "stfips.feather"))
# catboost.save_model(model, "models/model.cbm")
# train_data_model_input <- train_data %>% select(-freq1_d)
# 
# features_to_use <- c("services_d", "hlthins", "los_binned", "stfips")  # add more if needed
# 
# train_data_subset <- train_data %>%
#   select(all_of(features_to_use)) %>%
#   mutate(across(everything(), ~ as.character(.)))  # ensure compatibility
# 
# write_feather(train_data_subset, "data/X_train_reduced.feather")
# 
# 
# 
# str(train_data[, c("services_d", "hlthins", "los_binned", "stfips")])
