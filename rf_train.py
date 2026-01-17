import pandas as pd
import numpy as np
import pickle
# preprocessing
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
# regression model
from sklearn.ensemble import RandomForestRegressor
# metrices
from sklearn.metrics import r2_score,mean_squared_error


# read dataset
df = pd.read_csv("./notebook/data/data.csv")

# drop column
df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'], errors='ignore', inplace=True)

# target and feature
X = df.drop("price",axis=1)
y=df['price']

# column separation
numeric_features = X.select_dtypes(include=np.number).columns
categorical_features = X.select_dtypes(exclude=np.number).columns


# Preprocessing
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, numeric_features),
    ('cat', cat_transformer, categorical_features)
])

# random forest model
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)

# pipeline
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', rf_model)
])

# train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train model
rf_pipeline.fit(X_train, y_train)

# evaluation
y_pred = rf_pipeline.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")


# Save model (IMPORTANT)
with open("laptop_rf_pipeline.pkl", "wb") as f:
    pickle.dump(rf_pipeline, f)

print("Random Forest pipeline saved as laptop_rf_pipeline.pkl")