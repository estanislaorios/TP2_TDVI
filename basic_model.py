'''
import pandas as pd
import gc
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Print all columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

# Load and combine the CSV files
csv_files = ["ctr_15.csv", "ctr_16.csv", "ctr_17.csv", "ctr_18.csv", "ctr_19.csv", "ctr_20.csv", "ctr_21.csv"]
combined_data = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

# Load the test data
eval_data = pd.read_csv("ctr_test.csv")

# Split the combine data into train and validation sets with a fixed random state
random_seed = 2345
combined_data = combined_data.sample(frac=8/10, random_state=random_seed)
y = combined_data["Label"]
X = combined_data.drop(columns=["Label"])
X = X.select_dtypes(include='number')

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=random_seed)

del combined_data, X, y
gc.collect()

# Train the model using a Decision Tree
cls = make_pipeline(SimpleImputer(), DecisionTreeClassifier(max_depth=8, random_state=random_seed))
cls.fit(X_train, y_train)

# Evaluate the model on the validation set
y_val_preds = cls.predict_proba(X_val)[:, cls.classes_ == 1].squeeze()
val_auc_roc = roc_auc_score(y_val, y_val_preds)
print(f'Validation AUC-ROC: {val_auc_roc:.4f}')

# Predict on the evaluation set
eval_data = eval_data.select_dtypes(include='number')
y_preds = cls.predict_proba(eval_data.drop(columns=["id"]))[:, cls.classes_ == 1].squeeze()

# Make the submission file
submission_df = pd.DataFrame({"id": eval_data["id"], "Label": y_preds})
submission_df["id"] = submission_df["id"].astype(int)
submission_df.to_csv("basic_model.csv", sep=",", index=False)
'''

import pandas as pd
import gc
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Print all columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

# Load and combine the CSV files
csv_files = ["ctr_15.csv", "ctr_16.csv", "ctr_17.csv", "ctr_18.csv", "ctr_19.csv", "ctr_20.csv", "ctr_21.csv"]
combined_data = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

# Load the test data
eval_data = pd.read_csv("ctr_test.csv")

# Feature Engineering: Add 'ad_size' (creative_height * creative_width)
combined_data['ad_size'] = combined_data['creative_height'] * combined_data['creative_width']
eval_data['ad_size'] = eval_data['creative_height'] * eval_data['creative_width']

# Split the combined data into train and validation sets with a fixed random state
random_seed = 2345
combined_data = combined_data.sample(frac=8/10, random_state=random_seed)
y = combined_data["Label"]
X = combined_data.drop(columns=["Label"])
X = X.select_dtypes(include='number')

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=random_seed)

del combined_data, X, y
gc.collect()

# Train the model using a Decision Tree
cls = make_pipeline(SimpleImputer(), DecisionTreeClassifier(max_depth=8, random_state=random_seed))
cls.fit(X_train, y_train)

# Evaluate the model on the validation set
y_val_preds = cls.predict_proba(X_val)[:, cls.classes_ == 1].squeeze()
val_auc_roc = roc_auc_score(y_val, y_val_preds)
print(f'Validation AUC-ROC: {val_auc_roc:.4f}')

# Predict on the evaluation set
eval_data = eval_data.select_dtypes(include='number')
y_preds = cls.predict_proba(eval_data.drop(columns=["id"]))[:, cls.classes_ == 1].squeeze()

# Make the submission file
submission_df = pd.DataFrame({"id": eval_data["id"], "Label": y_preds})
submission_df["id"] = submission_df["id"].astype(int)
submission_df.to_csv("basic_model.csv", sep=",", index=False)
