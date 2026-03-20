# ==========================================
# SALES PREDICTION SYSTEM COMPLETE PROJECT
# ==========================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("train.csv")

# ==========================================
# CHECK DATA
# ==========================================

print("First 5 Rows:")
print(df.head())

print("\nColumns:")
print(df.columns)

# Remove extra spaces
df.columns = df.columns.str.strip()

# ==========================================
# HANDLE MISSING VALUES
# ==========================================

df.fillna(method='ffill', inplace=True)

# ==========================================
# DATE CONVERSION
# ==========================================

df['Order Date'] = pd.to_datetime(
    df['Order Date'],
    dayfirst=True,
    errors='coerce'
)

# Extract month and year
df['Month'] = df['Order Date'].dt.month
df['Year'] = df['Order Date'].dt.year

# ==========================================
# ENCODE CATEGORICAL DATA
# ==========================================

le = LabelEncoder()

df['Category'] = le.fit_transform(df['Category'])
df['Region'] = le.fit_transform(df['Region'])

# ==========================================
# FEATURE SELECTION
# ==========================================

X = df[['Category', 'Region', 'Month']]
y = df['Sales']

# ==========================================
# FEATURE SCALING
# ==========================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# ==========================================
# LINEAR REGRESSION
# ==========================================

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

print("\nLinear Regression Results")
print("R2 Score:", r2_score(y_test, y_pred_lr))
print("MSE:", mean_squared_error(y_test, y_pred_lr))

# ==========================================
# RANDOM FOREST REGRESSION
# ==========================================

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("\nRandom Forest Results")
print("R2 Score:", r2_score(y_test, y_pred_rf))
print("MSE:", mean_squared_error(y_test, y_pred_rf))

# ==========================================
# EDA VISUALIZATION 1 : SALES DISTRIBUTION
# ==========================================

plt.figure(figsize=(8,5))
plt.hist(df['Sales'], bins=30)
plt.title("Sales Distribution")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.show()

# ==========================================
# EDA VISUALIZATION 2 : MONTHLY SALES TREND
# ==========================================

monthly_sales = df.groupby('Month')['Sales'].sum()

plt.figure(figsize=(8,5))
plt.plot(monthly_sales)
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.show()

# ==========================================
# EDA VISUALIZATION 3 : CATEGORY SALES
# ==========================================

df.groupby('Category')['Sales'].sum().plot(kind='bar')
plt.title("Category Wise Sales")
plt.show()

# ==========================================
# HEATMAP
# ==========================================

numeric_df = df.select_dtypes(include='number')

plt.figure(figsize=(8,6))
sns.heatmap(numeric_df.corr(), annot=True)
plt.show()

# ==========================================
# ACTUAL VS PREDICTED
# ==========================================

plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred_rf)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")

plt.show()

# ==========================================
# SIMPLE DASHBOARD STYLE OUTPUT
# ==========================================

print("\nTotal Sales:", df['Sales'].sum())
print("Average Sales:", df['Sales'].mean())
print("Maximum Sales:", df['Sales'].max())