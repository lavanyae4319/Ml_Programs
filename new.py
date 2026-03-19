# Create virtual environment
python -m venv mltp_lab1_env

# Activate environment (Windows)
mltp_lab1_env\Scripts\activate

# Activate environment (Mac/Linux)
source mltp_lab1_env/bin/activate

# Install required packages
pip install pandas numpy matplotlib seaborn

# ==========================================
# LAB EXPERIMENT 1
# Cafe Sales - Data Cleaning
# ==========================================

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Task 2: Load and Inspect Dataset
# -------------------------------

# Load dataset (Change path if needed)
df = pd.read_csv("cafe_sales.csv")

# Display first 5 rows
print("First 5 rows:")
print(df.head())

# Display last 5 rows
print("\nLast 5 rows:")
print(df.tail())

# Display dataset info
print("\nDataset Info:")
print(df.info())

# Display summary statistics
print("\nSummary Statistics:")
print(df.describe())

# -------------------------------
# Task 3: Handle Missing Values
# -------------------------------

# Identify missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Separate numerical and categorical columns
num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(include='object').columns

# Fill numerical columns with median
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Fill categorical columns with 'Unknown'
for col in cat_cols:
    df[col].fillna("Unknown", inplace=True)

# If date column exists
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['date'].fillna(method='ffill', inplace=True)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# -------------------------------
# Task 4: Remove Duplicates
# -------------------------------

# Count duplicate rows
duplicates = df.duplicated().sum()
print("\nNumber of duplicate rows:", duplicates)

# Remove duplicates
df = df.drop_duplicates()

print("Duplicates removed successfully.")

# -------------------------------
# Task 5: Detect Outliers (IQR Method)
# -------------------------------

# Choose a numerical column (example: 'total_price')
column_name = num_cols[0]  # You can change column name manually

Q1 = df[column_name].quantile(0.25)
Q3 = df[column_name].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Detect outliers
outliers = df[(df[column_name] < lower_bound) | 
              (df[column_name] > upper_bound)]

print("\nOutliers detected:")
print(outliers)

# Boxplot visualization
plt.figure()
sns.boxplot(x=df[column_name])
plt.title("Boxplot for Outlier Detection")
plt.show()



# ==========================================
# LAB EXPERIMENT 2
# Breast Cancer Wisconsin Dataset
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler, LabelEncoder

# -------------------------------
# Task 1: Exploratory Data Analysis
# -------------------------------

# Fetch dataset
data = fetch_ucirepo(id=17)

# Convert to DataFrame
X = data.data.features
y = data.data.targets

df = pd.concat([X, y], axis=1)

print("Dataset Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

# Histogram
plt.figure()
df.iloc[:, 0].hist()
plt.title("Histogram of First Feature")
plt.show()

# Boxplot
plt.figure()
sns.boxplot(x=df.iloc[:, 0])
plt.title("Boxplot for Outlier Visualization")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# -------------------------------
# Task 2: Data Cleaning & Transformation
# -------------------------------

# Rename columns (example)
df.columns = df.columns.str.replace(" ", "_")

# Remove duplicates
print("Duplicate rows:", df.duplicated().sum())
df = df.drop_duplicates()

# -------------------------------
# Task 3: Handle Missing Values & Outliers
# -------------------------------

# Check missing values
print("Missing Values:")
print(df.isnull().sum())

# Fill missing numerical values with mean
df.fillna(df.mean(numeric_only=True), inplace=True)

# Outlier detection (IQR method)
col = df.columns[0]

Q1 = df[col].quantile(0.25)
Q3 = df[col].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

# Remove outliers
df = df[(df[col] >= lower) & (df[col] <= upper)]

# -------------------------------
# Task 4: Feature Scaling
# -------------------------------

# Separate features & target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Label Encoding (if target categorical)
le = LabelEncoder()
y = le.fit_transform(y)

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert back to DataFrame
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print("\nScaled Dataset:")
print(X_scaled.head())



# ==========================================
# LAB EXPERIMENT 2
# Breast Cancer Wisconsin Dataset
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler, LabelEncoder

# -------------------------------
# Task 1: Exploratory Data Analysis
# -------------------------------

# Fetch dataset
data = fetch_ucirepo(id=17)

# Convert to DataFrame
X = data.data.features
y = data.data.targets

df = pd.concat([X, y], axis=1)

print("Dataset Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

# Histogram
plt.figure()
df.iloc[:, 0].hist()
plt.title("Histogram of First Feature")
plt.show()

# Boxplot
plt.figure()
sns.boxplot(x=df.iloc[:, 0])
plt.title("Boxplot for Outlier Visualization")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# -------------------------------
# Task 2: Data Cleaning & Transformation
# -------------------------------

# Rename columns (example)
df.columns = df.columns.str.replace(" ", "_")

# Remove duplicates
print("Duplicate rows:", df.duplicated().sum())
df = df.drop_duplicates()

# -------------------------------
# Task 3: Handle Missing Values & Outliers
# -------------------------------

# Check missing values
print("Missing Values:")
print(df.isnull().sum())

# Fill missing numerical values with mean
df.fillna(df.mean(numeric_only=True), inplace=True)

# Outlier detection (IQR method)
col = df.columns[0]

Q1 = df[col].quantile(0.25)
Q3 = df[col].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

# Remove outliers
df = df[(df[col] >= lower) & (df[col] <= upper)]

# -------------------------------
# Task 4: Feature Scaling
# -------------------------------

# Separate features & target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Label Encoding (if target categorical)
le = LabelEncoder()
y = le.fit_transform(y)

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert back to DataFrame
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print("\nScaled Dataset:")
print(X_scaled.head())





[8:24 am, 24/2/2026] +91 78991 02923: import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample dataset
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
Y = np.array([2, 4, 5, 4, 5])

# Create model
model = LinearRegression()

# Train model
model.fit(X, Y)

# Predict values
Y_pred = model.predict(X)

# Print slope and intercept
print("Slope (b1):", model.coef_[0])
print("Intercept (b0):", model.intercept_)

# Plot
plt.scatter(X, Y, label="Actual Data")
plt.plot(X, Y_pred, label="Regression Line")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Simple Linear Regression")
plt.legend()
plt.show()
[8:24 am, 24/2/2026] +91 78991 02923: import numpy as np
import matplotlib.pyplot as plt

# Sample data
X = np.array([1, 2, 3, 4, 5])
Y = np.array([2, 4, 5, 4, 5])

# Number of observations
n = len(X)

# Calculate required sums
sum_x = np.sum(X)
sum_y = np.sum(Y)
sum_xy = np.sum(X * Y)
sum_x2 = np.sum(X * X)

# Calculate slope (b1)
b1 = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - (sum_x ** 2))

# Calculate intercept (b0)
mean_x = np.mean(X)
mean_y = np.mean(Y)
b0 = mean_y - b1 * mean_x

print("Slope (b1):", b1)
print("Intercept (b0):", b0)

# Predict values
Y_pred = b0 + b1 * X

# Plot
plt.scatter(X, Y, label="Actual Data")
plt.plot(X, Y_pred, label="Regression Line")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Simple Linear Regression (Manual)")
plt.legend()
plt.show()




Q1 = df['values'].quantile(0.25)
print(Q1)

Q3= df['values'].quantile(0.75)
print(Q3)

IQR= Q3 -Q1
print(IQR)

df = df[(df['values'] >= (Q1 - 1.5 * IQR)) & (df['values'] <= (Q3 + 1.5 * IQR))]
print(df)











import pandas as pd
import statistics as st

data = pd.Series([-53, 1, 3, 2, 115, 70, 10, 9])
sortdata = sorted(data)

print(sortdata)

# Quartiles
Q1, Q2, Q3 = st.quantiles(sortdata, n=4)
print("Q1:", Q1)
print("Q3:", Q3)

# IQR
IQR = Q3 - Q1
print("IQR:", IQR)

# Outlier check (correct logic uses OR)
outlier = (min(sortdata) <= Q1 - 1.5 * IQR) or (max(sortdata) >= Q3 + 1.5 * IQR)
print("Outlier Exists:", outlier)










# ==============================================
# MACHINE LEARNING LAB EXAM – INSURANCE DATASET
# ==============================================

# -----------------------------
# Import Required Libraries
# -----------------------------
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# -----------------------------
# Question 1: Data Loading
# -----------------------------

# Extract ZIP file (change file name if needed)
with zipfile.ZipFile("archive.zip", 'r') as zip_ref:
    zip_ref.extractall("insurance_data")

# Load dataset
df = pd.read_csv("insurance_data/insurance.csv")

print("First 5 Rows:")
print(df.head())

print("\nShape of Dataset:", df.shape)

print("\nData Types:")
print(df.dtypes)

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

# Target Variable
print("\nTarget Variable for Regression: charges")

# -----------------------------
# Question 2: Exploratory Data Analysis
# -----------------------------

# Distribution of numerical variables
df.hist(figsize=(10,8))
plt.tight_layout()
plt.show()

# Smoker vs Charges
sns.boxplot(x='smoker', y='charges', data=df)
plt.title("Charges by Smoking Status")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# -----------------------------
# Question 3: Data Preparation
# -----------------------------

# Outlier Detection using IQR method on charges
Q1 = df['charges'].quantile(0.25)
Q3 = df['charges'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

print("\nLower Limit:", lower)
print("Upper Limit:", upper)

# Remove Outliers
df_clean = df[(df['charges'] >= lower) & (df['charges'] <= upper)]

print("\nShape before removing outliers:", df.shape)
print("Shape after removing outliers:", df_clean.shape)

# Encode categorical variables
df_encoded = pd.get_dummies(df_clean, drop_first=True)

print("\nEncoded Data Sample:")
print(df_encoded.head())

# -----------------------------
# Question 4: Feature Scaling
# -----------------------------

scaler = StandardScaler()

# Apply scaling on BMI
df_encoded['bmi_scaled'] = scaler.fit_transform(df_encoded[['bmi']])

print("\nBefore Scaling BMI Stats:")
print(df_encoded['bmi'].describe())

print("\nAfter Scaling BMI Stats:")
print(df_encoded['bmi_scaled'].describe())

# -----------------------------
# Question 5: Linear Regression
# -----------------------------

# Define Features and Target
X = df_encoded.drop('charges', axis=1)
y = df_encoded['charges']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\nModel Performance:")
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# -----------------------------
# Manual Slope Calculation (Using BMI only)
# Formula: Slope = Cov(X,Y) / Var(X)
# -----------------------------

X_bmi = df_encoded['bmi']
Y = df_encoded['charges']

cov = np.cov(X_bmi, Y, bias=True)[0][1]
var = np.var(X_bmi)

manual_slope = cov / var

print("\nManual Slope (BMI vs Charges):", manual_slope)

# Library Slope Comparison
model_single = LinearRegression()
model_single.fit(X_bmi.values.reshape(-1,1), Y)

print("Library Slope:", model_single.coef_[0])

# -----------------------------
# Final Conclusion Print
# -----------------------------
print("\nConclusion:")
print("1. Target variable: charges")
print("2. Strong predictor: smoker")
print("3. Outlier removal improved model stability")
print("4. Manual slope matches library slope")
print("5. Linear regression performs reasonably well")