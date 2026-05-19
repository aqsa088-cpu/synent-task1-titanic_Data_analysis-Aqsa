# 🚢 Titanic Dataset — Data Preprocessing

> **Task 1 of the SynEnt ML Pipeline** | Data Cleaning & Feature Engineering

---

## Problem Statement

The Titanic dataset is a classic binary classification benchmark where the goal is to **predict passenger survival** based on sociodemographic and travel attributes. However, raw tabular data from real-world sources is rarely model-ready — it contains missing values, inconsistent types, redundant features, and poorly named columns.

This task addresses the **data preprocessing stage**: transforming the raw Titanic CSV into a clean, numerically encoded, and well-structured DataFrame that is ready for downstream model training. Without this step, any machine learning model would produce unreliable or invalid results.

---

## Dataset Details

| Property | Value |
|---|---|
| **Source** | Titanic-Dataset.csv (from `titanicDataset.zip`) |
| **Total Records** | 891 passengers |
| **Target Variable** | `Survived` (0 = No, 1 = Yes) |
| **Original Features** | 12 columns |
| **Final Features** | 11 columns (post-processing) |

### Original Columns

| Column | Type | Description |
|---|---|---|
| `PassengerId` | int | Unique passenger identifier |
| `Survived` | int | Survival label (target) |
| `Pclass` | int | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd) |
| `Name` | str | Passenger name |
| `Sex` | str | Gender |
| `Age` | float | Age in years |
| `SibSp` | int | # of siblings / spouses aboard |
| `Parch` | int | # of parents / children aboard |
| `Ticket` | str | Ticket number |
| `Fare` | float | Passenger fare |
| `Cabin` | str | Cabin number (**687 missing**) |
| `Embarked` | str | Port of embarkation (C / Q / S) — **2 missing** |

### Missing Values Summary (Pre-processing)

| Column | Missing Count | % Missing |
|---|---|---|
| `Cabin` | 687 | 77.1% |
| `Age` | 177 | 19.9% |
| `Embarked` | 2 | 0.2% |

---

## Approach

The preprocessing pipeline follows six sequential steps:

### 1. Data Loading
The dataset is extracted from a ZIP archive and loaded into a pandas DataFrame. Initial inspection (`head()`, `info()`) confirms column types, shape, and the presence of nulls.

### 2. Handling Missing Values

| Column | Strategy | Justification |
|---|---|---|
| `Age` | Impute with **median** | Median is robust to skew; preserves row count |
| `Cabin` | **Drop column** | 77% missing — too sparse to impute meaningfully |
| `Embarked` | Impute with **mode** | Only 2 missing; majority class fill is appropriate |

### 3. Encoding Categorical Features
`Sex` and `Embarked` are converted to numerical form using **one-hot encoding** (`pd.get_dummies` with `drop_first=True` to avoid multicollinearity):

- `Sex` → `Sex_male` (0 = female, 1 = male)
- `Embarked` → `Embarked_Q`, `Embarked_S` (C is the dropped reference category)

### 4. Removing Duplicate Rows
Duplicate records are identified and dropped to prevent data leakage and biased model training.

### 5. Converting Data Types
Boolean columns produced by one-hot encoding (`Sex_male`, `Embarked_Q`, `Embarked_S`) are cast to `int` (0/1) for compatibility with scikit-learn and other ML libraries.

### 6. Renaming Columns
Cryptic column names are replaced with descriptive equivalents for improved code readability:

| Original | Renamed |
|---|---|
| `Pclass` | `Passenger_Class` |
| `SibSp` | `Siblings_Spouses` |
| `Parch` | `Parents_Children` |

---

## Results

After preprocessing, the dataset is clean and model-ready:

| Metric | Before | After |
|---|---|---|
| **Columns** | 12 | 11 |
| **Missing Values** | 866 total | **0** |
| **Categorical Columns** | 2 (`Sex`, `Embarked`) | 0 (encoded) |



| **Boolean Columns** | 0 | 0 (converted to int) |
| **Duplicate Rows** | Checked | Removed |
