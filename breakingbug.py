
# Import libraries

# 1. To handle the data
import pandas as pd
import numpy as np

# 2. To visualize the data
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from yellowbrick.cluster import KElbowVisualizer
from matplotlib.colors import ListedColormap

# 3. To preprocess the data
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer

# 4. Import Iterative imputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# 5. Machine Learning
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

# 6. For Classification tasks
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB

# 7. Metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 8. Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv("/kaggle/input/heart-disease-data/heart_disease_uci.csv")

# Print the first 5 rows of the dataframe
print(df.head())

# Exploring the data type of each column
print(df.info())

# Checking the data shape
print(df.shape)

# Id column
print('Id column:', df['id'].min(), df['id'].max())

# Age column
print('Age column:', df['age'].min(), df['age'].max())

# Summarize the age column
print(df['age'].describe())

# Define custom colors
custom_colors = ["#FF5733", "#3366FF", "#33FF57"]  # Example colors, you can adjust as needed

# Plot the histogram with custom colors
sns.histplot(df['age'], kde=True, color="#FF5733")
plt.show()

# Plot the mean, median, and mode of the age column using sns
sns.histplot(df['age'], kde=True)
plt.axvline(df['age'].mean(), color='Red', linestyle='--', label='Mean')
plt.axvline(df['age'].median(), color='Green', linestyle='--', label='Median')
plt.axvline(df['age'].mode()[0], color='Blue', linestyle='--', label='Mode')
plt.legend()
plt.show()

# Print the value of mean, median, and mode of age column
print('Mean:', df['age'].mean())
print('Median:', df['age'].median())
print('Mode:', df['age'].mode()[0])

# Plot the histogram of age column using plotly and color by sex
fig = px.histogram(data_frame=df, x='age', color='sex')
fig.show()

# Find the values of the sex column
print('Sex value counts:')
print(df['sex'].value_counts())

# Calculating the percentage of male and female value counts in the data
male_count = df['sex'].value_counts().get('male', 0)
female_count = df['sex'].value_counts().get('female', 0)

total_count = male_count + female_count

# Calculate percentages
male_percentage = (male_count / total_count) * 100 if total_count > 0 else 0
female_percentage = (female_count / total_count) * 100 if total_count > 0 else 0

# Display the results
print(f'Male percentage in the data: {male_percentage:.2f}%')
print(f'Female percentage in the data: {female_percentage:.2f}%')

# Difference
difference_percentage = ((male_count - female_count) / female_count * 100) if female_count > 0 else 0
print(f'Males are {difference_percentage:.2f}% more than females in the data.')

# Find the value counts of the age column grouped by the sex column
print(df.groupby('sex')['age'].value_counts())

# Find the unique values in the dataset column
print(df['dataset'].value_counts())  # Corrected: use value_counts() instead of counts()

# Plot the countplot of the dataset column
fig = px.bar(df, x='dataset', color='sex')
fig.show()

# Print the values of the dataset column grouped by sex
print(df.groupby('sex')['dataset'].value_counts())

# Make a plot of the age column using plotly and coloring by dataset
fig = px.histogram(data_frame=df, x='age', color='dataset')
fig.show()

# Print the mean, median, and mode of the age column grouped by the dataset column
print("___________________________________________________________")
print("Mean of the dataset: ", df.groupby('dataset')['age'].mean())
print("___________________________________________________________")
print("Median of the dataset: ", df.groupby('dataset')['age'].median())
print("___________________________________________________________")
print("Mode of the dataset: ", df.groupby('dataset')['age'].apply(pd.Series.mode))
print("___________________________________________________________")

# Value count of cp column
print(df['cp'].value_counts())

# Count plot of cp column by sex column
sns.countplot(data=df, x='cp', hue='sex')
plt.show()

# Count plot of cp column by dataset column
sns.countplot(data=df, x='cp', hue='dataset')
plt.show()

# Draw the plot of the age column grouped by the cp column
fig = px.histogram(data_frame=df, x='age', color='cp')
fig.show()

# Summarize the trestbps column
print(df['trestbps'].describe())

# Dealing with missing values in the trestbps column
# Find the percentage of missing values in the trestbps column
print(f"Percentage of missing values in trestbps column: {df['trestbps'].isnull().sum() / len(df) * 100:.2f}%")

# Impute the missing values of the trestbps column using iterative imputer
# Create an object of IterativeImputer
imputer1 = IterativeImputer(max_iter=10, random_state=42)

# Fit the imputer on the trestbps column
imputer1.fit(df[['trestbps']])

# Transform the data
df['trestbps'] = imputer1.transform(df[['trestbps']])

# Check the missing values in the trestbps column
print(f"Missing values in trestbps column after imputation: {df['trestbps'].isnull().sum()}")

# First let's see data types or categories of columns
print(df.info())

# Let's see which columns have missing values
print((df.isnull().sum() / len(df) * 100).sort_values(ascending=False))

# Create an object of IterativeImputer
imputer2 = IterativeImputer(max_iter=10, random_state=42)

# Fit and transform on ca, oldpeak, thal, chol, and thalch columns
# Correcting the method of imputation
df['ca'] = imputer2.fit_transform(df[['ca']])
df['oldpeak'] = imputer2.fit_transform(df[['oldpeak']])
df['chol'] = imputer2.fit_transform(df[['chol']])
df['thal'] = imputer2.fit_transform(df[['thal']])
df['thalch'] = imputer2.fit_transform(df[['thalch']])




# Let's check again for missing values
print((df.isnull().sum() / len(df) * 100).sort_values(ascending=False))

print(f"The missing values in 'thal' column are: {df['thal'].isnull().sum()}")

print(df['thal'].value_counts())

print(df.tail())

# Find missing values (corrected method)
missing_data_cols = df.isnull().sum()[df.isnull().sum() > 0].index.tolist()
print(f"Columns with missing values: {missing_data_cols}")

# Find categorical columns
cat_cols = df.select_dtypes(include='object').columns.tolist()
print(f'Categorical Columns: {cat_cols}')

# Find numerical columns
num_cols = df.select_dtypes(exclude='object').columns.tolist()
print(f'Numerical Columns: {num_cols}')

# Define columns
categorical_cols = ['thal', 'ca', 'slope', 'exang', 'restecg', 'thalch', 'chol', 'trestbps']
bool_cols = ['fbs']
numerical_cols = ['oldpeak', 'age', 'restecg', 'fbs', 'cp', 'sex', 'num']

# Function to impute missing values in categorical columns
def impute_categorical_missing_data(passed_col):
    df_null = df[df[passed_col].isnull()]
    df_not_null = df[df[passed_col].notnull()]

    X = df_not_null.drop(passed_col, axis=1)
    y = df_not_null[passed_col]

    other_missing_cols = [col for col in missing_data_cols if col != passed_col]

    label_encoder = LabelEncoder()

    # Encoding categorical columns
    for col in y.columns:
        if y[col].dtype == 'object':
            y[col] = label_encoder.fit_transform(y[col].astype(str))

    if passed_col in bool_cols:
        y = label_encoder.fit_transform(y)

    # Creating and fitting the imputer
    imputer = IterativeImputer(estimator=RandomForestRegressor(random_state=16), add_indicator=True)
    
    # Impute missing values for other columns
    for col in other_missing_cols:
        cols_with_missing_value = X[col].values.reshape(-1, 1)
        imputed_values = imputer.fit_transform(cols_with_missing_value)
        X[col] = imputed_values[:, 0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_classifier = RandomForestClassifier()

    rf_classifier.fit(X_train, y_train)

    y_pred = rf_classifier.predict(X_test)

    acc_score = accuracy_score(y_test, y_pred)

    print(f"The feature '{passed_col}' has been imputed with", round((acc_score * 100), 2), "% accuracy\n")

    X = df_null.drop(passed_col, axis=1)

    # Encoding categorical columns again
    for col in df_null.columns:
        if df[col].dtype == 'object':
            df[col] = label_encoder.fit_transform(df[col].astype(str))

    for col in other_missing_cols:
        cols_with_missing_value = df[col].values.reshape(-1, 1)
        imputed_values = imputer.fit_transform(cols_with_missing_value)
        X[col] = imputed_values[:, 0]

    if len(df_null) > 0:
        df[passed_col] = rf_classifier.predict(X)
        if passed_col in bool_cols:
            df[passed_col] = df[passed_col].map({0: False, 1: True})

    df_combined = pd.concat([df_not_null, df_null])

    return df_combined[passed_col]

from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def impute_continuous_missing_data(passed_col):
    df_null = df[df[passed_col].isnull()]
    df_not_null = df[df[passed_col].notnull()]

    X = df_not_null.drop(passed_col, axis=1)
    y = df_not_null[passed_col]

    other_missing_cols = [col for col in missing_data_cols if col != passed_col]

    # Label Encoding for categorical features in the training data
    label_encoder = LabelEncoder()

    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = label_encoder.fit_transform(X[col].astype(str))

    # Creating and fitting the imputer
    imputer = IterativeImputer(estimator=RandomForestRegressor(random_state=16), add_indicator=True)

    # Impute missing values for other columns
    for col in other_missing_cols:
        if X[col].isnull().any():
            cols_with_missing_value = X[col].values.reshape(-1, 1)
            imputed_values = imputer.fit_transform(cols_with_missing_value)
            X[col] = imputed_values[:, 0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_regressor = RandomForestRegressor()

    rf_regressor.fit(X_train, y_train)

    y_pred = rf_regressor.predict(X_test)

    print(f"MAE = {mean_absolute_error(y_test, y_pred)}")
    print(f"RMSE = {mean_squared_error(y_test, y_pred, squared=False)}")
    print(f"R2 = {r2_score(y_test, y_pred)}")

    X = df_null.drop(passed_col, axis=1)

    # Label Encoding for categorical features in the prediction data
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = label_encoder.fit_transform(X[col].astype(str))

    for col in other_missing_cols:
        if X[col].isnull().any():
            cols_with_missing_value = X[col].values.reshape(-1, 1)
            imputed_values = imputer.fit_transform(cols_with_missing_value)
            X[col] = imputed_values[:, 0]

    if len(df_null) > 0:
        df[passed_col] = rf_regressor.predict(X)

    df_combined = pd.concat([df_not_null, df_null])

    return df_combined[passed_col]

# Check for missing values
print(df.isnull().sum().sort_values(ascending=False))

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Impute missing values using our functions
for col in missing_data_cols:
    print(f"Missing Values in {col}: {round((df[col].isnull().sum() / len(df)) * 100, 2)}%")
    if col in categorical_cols:
        df[col] = impute_categorical_missing_data(col)
    elif col in numerical_cols:
        df[col] = impute_continuous_missing_data(col)

# Check again for missing values
print(df.isnull().sum().sort_values(ascending=False))


import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Print separator line
print("_________________________________________________________________________________________________________________________________________________")

# Set seaborn style and color palette
sns.set(rc={"axes.facecolor":"#87CEEB","figure.facecolor":"#EEE8AA"})  # Change figure background color
palette = ["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"]
cmap = ListedColormap(palette)

# Plot boxenplots for columns
plt.figure(figsize=(10,8))
for i, col in enumerate(cols):
    plt.subplot(3, 2, i + 1)  # Corrected subplot index to start at 1
    sns.boxenplot(data=df, x=col, color=palette[i % len(palette)])  # Use modulo to cycle through colors
    plt.title(col)

plt.tight_layout()  # Adjust subplots to fit into figure area
plt.show()

# Print rows where trestbps value is 0
print("Rows with trestbps = 0:")
print(df[df['trestbps'] == 0])

# Remove outlier rows where trestbps = 0
df = df[df['trestbps'] != 0]

# Modify seaborn style and color palette
sns.set(rc={"axes.facecolor":"#B76E79","figure.facecolor":"#C0C0C0"})
modified_palette = ["#C44D53", "#B76E79", "#DDA4A5", "#B3BCC4", "#A2867E", "#F3AB60"]
cmap = ListedColormap(modified_palette)

# Plot boxenplots with modified color palette
plt.figure(figsize=(10,8))
for i, col in enumerate(cols):
    plt.subplot(3, 2, i + 1)  # Corrected subplot index to start at 1
    sns.boxenplot(data=df, x=col, color=modified_palette[i % len(modified_palette)])
    plt.title(col)

plt.tight_layout()  # Adjust subplots to fit into figure area
plt.show()

# Print description of trestbps column
print("Description of trestbps column:")
print(df['trestbps'].describe())

# Print description of the entire dataframe
print("Description of the dataframe:")
print(df.describe())

print("___________________________________________________________________________________________________________________________________________________________________")

# Set facecolors for seaborn plots
sns.set(rc={"axes.facecolor": "#FFF9ED", "figure.facecolor": "#FFF9ED"})

# Define the "night vision" color palette
night_vision_palette = ["#00FF00", "#FF00FF", "#00FFFF", "#FFFF00", "#FF0000", "#0000FF"]

# Plot boxenplots with the "night vision" palette
plt.figure(figsize=(10, 8))
for i, col in enumerate(cols):
    plt.subplot(3, 2, i + 1)  # Corrected subplot index to start at 1
    sns.boxenplot(data=df, x=col, color=night_vision_palette[i % len(night_vision_palette)])
    plt.title(col)

plt.tight_layout()  # Adjust subplots to fit into figure area
plt.show()

# Plot histogram of trestbps column
palette = ["#999999", "#666666", "#333333"]
sns.histplot(data=df, x='trestbps', kde=True, color=palette[0])

plt.title('Resting Blood Pressure')
plt.xlabel('Pressure (mmHg)')
plt.ylabel('Count')

# Apply style and colors to the plot
plt.style.use('default')
plt.rcParams['figure.facecolor'] = palette[1]
plt.rcParams['axes.facecolor'] = palette[2]
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import warnings
warnings.filterwarnings('ignore')

# Create a histogram plot of trestbps column to analyze with sex column
sns.histplot(df, x='trestbps', kde=True, hue='sex', palette='Spectral')
plt.title('Distribution of Resting Blood Pressure by Sex')
plt.xlabel('Resting Blood Pressure (mmHg)')
plt.ylabel('Count')
plt.show()

# Print DataFrame information and first few rows
print(df.info())
print(df.columns)
print(df.head())

# Split the data into X and y
X = df.drop('num', axis=1)
y = df['num']

# Separate encoder for all categorical and object columns and inverse transform at the end
label_encoder = LabelEncoder()
onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

# Encode categorical columns
cat_cols = X.select_dtypes(include=['object']).columns
for col in cat_cols:
    X[col] = label_encoder.fit_transform(X[col].astype(str))

# Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Import all models
models = [
    ('Logistic Regression', LogisticRegression(random_state=42)),
    ('Gradient Boosting', GradientBoostingClassifier(random_state=42)),
    ('KNeighbors Classifier', KNeighborsClassifier()),
    ('Decision Tree Classifier', DecisionTreeClassifier(random_state=42)),
    ('AdaBoost Classifier', AdaBoostClassifier(random_state=42)),
    ('Random Forest', RandomForestClassifier(random_state=42)),
    ('XGBoost Classifier', XGBClassifier(random_state=42)),
    ('Support Vector Machine', SVC(random_state=42)),
    ('Naive Bayes Classifier', GaussianNB())
]

best_model = None
best_accuracy = 0.0

# Iterate over the models and evaluate their performance
for name, model in models:
    # Create a pipeline for each model
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('model', model)
    ])

    # Perform cross-validation
    scores = cross_val_score(pipeline, X_train, y_train, cv=5)
    mean_accuracy = scores.mean()

    # Fit the pipeline on the training data
    pipeline.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = pipeline.predict(X_test)

    # Calculate accuracy score
    accuracy = accuracy_score(y_test, y_pred)

    # Print performance metrics
    print("Model:", name)
    print("Cross-Validation Accuracy: {:.2f}".format(mean_accuracy))
    print("Test Accuracy: {:.2f}".format(accuracy))
    print()

    # Check if the current model has the best accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = pipeline

# Retrieve the best model
print("Best Model:", best_model)




import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')

categorical_cols = ['thal', 'ca', 'slope', 'exang', 'restecg', 'fbs', 'cp', 'sex', 'num']

def evaluate_classification_models(X, y, categorical_columns):
    # Encode categorical columns
    X_encoded = X.copy()
    label_encoders = {}
    for col in categorical_columns:
        if X[col].dtype == 'object':
            label_encoders[col] = LabelEncoder()
            X_encoded[col] = label_encoders[col].fit_transform(X[col].astype(str))

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(),
        "KNN": KNeighborsClassifier(),
        "NB": GaussianNB(),
        "SVM": SVC(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "AdaBoost": AdaBoostClassifier()
    }

    # Train and evaluate models
    results = {}
    best_model = None
    best_accuracy = 0.0
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = name

    return results, best_model

# Example usage:
X = df[categorical_cols]  # Select the categorical columns as input features
y = df['num']  # Select the target column

results, best_model = evaluate_classification_models(X, y, categorical_cols)
print("Model accuracies:", results)
print("Best model:", best_model)

def hyperparameter_tuning(X, y, categorical_columns, models):
    # Define dictionary to store results
    results = {}

    # Encode categorical columns
    X_encoded = X.copy()
    for col in categorical_columns:
        if X[col].dtype == 'object':
            label_encoders[col] = LabelEncoder()
            X_encoded[col] = label_encoders[col].fit_transform(X[col].astype(str))

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    # Perform hyperparameter tuning for each model
    for model_name, model in models.items():
        # Define parameter grid for hyperparameter tuning
        param_grid = {}
        if model_name == 'Logistic Regression':
            param_grid = {'C': [0.1, 1, 10, 100]}
        elif model_name == 'KNN':
            param_grid = {'n_neighbors': [3, 5, 7, 9]}
        elif model_name == 'NB':
            param_grid = {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]}
        elif model_name == 'SVM':
            param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.1, 1, 10, 100]}
        elif model_name == 'Decision Tree':
            param_grid = {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}
        elif model_name == 'Random Forest':
            param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}
        elif model_name == 'XGBoost':
            param_grid = {'learning_rate': [0.01, 0.1, 0.2], 'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7]}
        elif model_name == 'Gradient Boosting':
            param_grid = {'learning_rate': [0.01, 0.1, 0.2], 'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7]}
        elif model_name == 'AdaBoost':
            param_grid = {'learning_rate': [0.01, 0.1, 0.2], 'n_estimators': [50, 100, 200]}

        # Perform hyperparameter tuning using GridSearchCV
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        # Get best hyperparameters and evaluate on test set
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Store results in dictionary
        results[model_name] = {'best_params': best_params, 'accuracy': accuracy}

    return results

# Define models dictionary
models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "NB": GaussianNB(),
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier()
}

# Example usage:
results = hyperparameter_tuning(X, y, categorical_cols, models)
for model_name, result in results.items():
    print("Model:", model_name)
    print("Best hyperparameters:", result['best_params'])
    print("Accuracy:", result['accuracy'])
    print()
    
