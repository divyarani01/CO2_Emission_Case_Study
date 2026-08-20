# CO2 Emissions of Vehicles — EDA & Interpretable Regression

An end-to-end machine learning case study focused on understanding the key factors driving **vehicle CO2 emissions** and building an **interpretable regression model** that can support vehicle design and emissions-reduction decisions.

The project goes beyond prediction accuracy by addressing data quality, exploratory analysis, multicollinearity, feature engineering, categorical encoding, model selection, and coefficient-level interpretation.

---

## 📌 Business Problem

The objective is to help the automotive industry understand:

* Which vehicle characteristics have the strongest influence on CO2 emissions?
* How much does engine size contribute to emissions?
* Does vehicle class have a significant impact after controlling for engine size?
* Do manufacturer/brand effects remain after accounting for vehicle characteristics?
* Which vehicle design or fleet-composition changes could have the greatest emissions impact?

The primary focus is therefore **interpretability and actionable insights**, rather than simply maximizing predictive accuracy.

---

## 📊 Dataset

The dataset contains **7,385 vehicle records and 12 variables** before cleaning.

### Features

| Feature                            | Description                                      |
| ---------------------------------- | ------------------------------------------------ |
| `Make`                             | Vehicle manufacturer                             |
| `Model`                            | Vehicle model                                    |
| `Vehicle Class`                    | Vehicle category such as compact, SUV, van, etc. |
| `Engine Size(L)`                   | Engine displacement in litres                    |
| `Cylinders`                        | Number of engine cylinders                       |
| `Transmission`                     | Original transmission code                       |
| `Fuel Type`                        | Vehicle fuel type                                |
| `Fuel Consumption City (L/100 km)` | City fuel consumption                            |
| `Fuel Consumption Hwy (L/100 km)`  | Highway fuel consumption                         |
| `Fuel Consumption Comb (L/100 km)` | Combined fuel consumption                        |
| `Fuel Consumption Comb (mpg)`      | Combined fuel economy                            |
| `CO2 Emissions(g/km)`              | Target variable                                  |

The target variable ranges from **96 to 522 g/km**, with an average of approximately **251 g/km** after cleaning.

---

## 🔎 Project Workflow

```text
Raw Dataset
     │
     ▼
Data Inspection & Cleaning
     │
     ▼
Exploratory Data Analysis
     │
     ├── Univariate Analysis
     ├── Bivariate Analysis
     ├── Correlation Analysis
     ├── Multicollinearity / VIF
     └── Outlier Analysis
     │
     ▼
Feature Engineering & Selection
     │
     ├── Transmission Family
     ├── Engine Feature Analysis
     └── Fuel Consumption Redundancy Analysis
     │
     ▼
Train / Test Split
     │
     ▼
Categorical Encoding + Scaling
     │
     ▼
Ridge Regression with Cross-Validated Alpha
     │
     ▼
Model Evaluation
     │
     ├── R²
     ├── RMSE
     ├── MAE
     └── Residual Analysis
     │
     ▼
Coefficient Interpretation
     │
     ▼
Business Insights & Recommendations
```

---

## 🧹 Data Cleaning

The initial dataset contained:

* **7,385 rows**
* **12 columns**
* No missing values
* **1,104 duplicate rows (~14.9%)**
* A small number of `Model` values containing leading/trailing whitespace

The cleaning process:

1. Stripped whitespace from categorical columns.
2. Removed exact duplicate records.
3. Retained valid high-end vehicles and outliers because they represent genuine, policy-relevant vehicles.

After cleaning:

```text
6,281 rows × 12 columns
```

No additional correction was applied because the remaining data was considered valid and physically plausible.

---

## 📈 Exploratory Data Analysis

### Numerical Variables

The numerical predictors showed moderate right skewness, approximately **0.8–1.1**, with high-end observations corresponding largely to large/high-consumption vehicles.

These were treated as genuine observations rather than data errors.

### Target Variable

`CO2 Emissions(g/km)`:

```text
Mean   ≈ 251.2 g/km
Median = 246 g/km
Min    = 96 g/km
Max    = 522 g/km
Skew   ≈ 0.56
```

The target distribution was sufficiently well behaved that no transformation was required.

---

## 🔗 Correlation Analysis

All six numerical predictors showed strong relationships with CO2 emissions.

Approximate correlations with the target:

| Feature                     | Correlation with CO2 |
| --------------------------- | -------------------: |
| Fuel Consumption City       |                +0.92 |
| Fuel Consumption Hwy        |                +0.88 |
| Fuel Consumption Comb       |                +0.92 |
| Fuel Consumption Comb (mpg) |                -0.91 |
| Engine Size                 |                +0.85 |
| Cylinders                   |                +0.83 |

Fuel-consumption variables showed the strongest relationships with CO2 emissions, which is expected because fuel burned is directly related to CO2 produced.

---

## ⚠️ Multicollinearity Analysis

A major challenge in the dataset was **severe multicollinearity**.

For example:

```text
Fuel Consumption Comb (L/100 km) → VIF ≈ 75,434
Fuel Consumption City (L/100 km) → VIF ≈ 30,257
Fuel Consumption Hwy (L/100 km)  → VIF ≈ 10,500
Cylinders                         → VIF ≈ 79
Engine Size                       → VIF ≈ 55
```

The extremely high VIF values were not simply statistical accidents.

Combined fuel consumption is mathematically derived from city and highway consumption:

```text
Combined ≈ 0.55 × City + 0.45 × Highway
```

Engine size and cylinder count were also highly correlated, with approximately:

```text
r ≈ 0.93
```

This motivated careful feature engineering rather than blindly feeding every numerical feature into the regression model.

---

## 🛠️ Feature Engineering & Selection

### Transmission

The original `Transmission` variable combined:

* Transmission family
* Gear count

Gear count did not demonstrate a stable relationship with CO2 within transmission families.

Therefore:

```text
Transmission
      ↓
Transmission_Family
```

was created, while the raw transmission code was removed.

---

### Engine Size vs Cylinders

Three candidates were evaluated:

```text
Engine Size
Cylinders
Engine Size / Cylinders
```

`Engine Size` had the strongest relationship with CO2:

```text
Engine Size              r ≈ 0.855
Cylinders                r ≈ 0.835
Engine Size / Cylinder   r ≈ 0.589
```

Therefore, `Engine Size(L)` was retained as the more interpretable and actionable feature.

---

### Fuel Consumption

The four fuel-consumption variables contained essentially the same information.

Therefore, the final feature engineering process retained:

```text
Fuel Consumption Comb (L/100 km)
```

as the representative fuel-consumption variable during the broader feature-engineering stage.

---

## 🎯 Final Modeling Features

For the **interpretable business model**, fuel consumption was deliberately excluded.

The final model uses:

```text
Engine Size(L)
Make
Vehicle Class
Transmission_Family
Fuel Type
```

### Why exclude fuel consumption?

Fuel consumption is effectively a mediator:

```text
Vehicle Design
     ↓
Engine Size / Vehicle Class
     ↓
Fuel Consumption
     ↓
CO2 Emissions
```

Including fuel consumption would allow the model to explain emissions primarily through fuel burned rather than identifying the upstream vehicle-design levers that policymakers and manufacturers can act upon.

### Why exclude Model?

`Model` contains approximately **2,048 categories** and behaves almost like a vehicle identifier.

Target encoding was investigated but produced severe multicollinearity because many models had only one or two observations. The model therefore risked effectively memorizing the target.

The final approach dropped `Model` and used one-hot encoding for `Make`, which has only 42 categories.

---

## 🔀 Train-Test Split

An **80/20 train-test split** was used:

```text
Training samples: 5,024
Testing samples:  1,257
```

The split was performed **before fitting encoders** so that preprocessing was learned only from the training data, avoiding data leakage.

---

## 🔠 Encoding & Scaling

Categorical variables were one-hot encoded using:

```python
OneHotEncoder(
    drop="first",
    handle_unknown="ignore"
)
```

`Engine Size(L)` was standardized using `StandardScaler`.

The categorical variables were kept as 0/1 indicators so their Ridge coefficients could be interpreted relative to their baseline categories.

---

## 🤖 Model

### Ridge Regression

A Ridge regression model with cross-validated regularization strength was selected because residual multicollinearity remained between some meaningful vehicle characteristics.

```python
RidgeCV(
    alphas=np.logspace(-3, 3, 50),
    cv=5
)
```

The selected regularization parameter was:

```text
α = 0.1207
```

Ridge was preferred because it stabilizes coefficient estimates in the presence of correlated predictors while preserving interpretability.

---

## 📊 Model Performance

| Metric |      Train |           Test |
| ------ | ---------: | -------------: |
| R²     |     0.8727 |     **0.8721** |
| RMSE   | 21.09 g/km | **21.47 g/km** |
| MAE    | 15.70 g/km | **16.25 g/km** |

The test MAE is approximately:

```text
6.4% of average CO2 emissions
```

The near-identical train and test R² values indicate good generalization with no meaningful evidence of overfitting.

### Key Result

> The interpretable model explains approximately **87% of the variance in vehicle CO2 emissions** using only engine size, manufacturer, vehicle class, transmission family, and fuel type.

---

## 📌 Model Interpretability

The Ridge coefficients represent estimated changes in CO2 emissions while holding the other variables constant.

The standardized `Engine Size(L)` coefficient is approximately:

```text
+41 g/km per standard deviation
```

This confirms engine size as a significant and actionable driver even after controlling for manufacturer and vehicle class.

---

## 💡 Key Business Insights

The analysis identifies the strongest factors influencing emissions, approximately in this order:

### 1. Vehicle Class

Vehicle class is the strongest visible lever.

Compared with the compact baseline:

```text
Passenger Van → approximately +115 g/km
Cargo Van     → approximately +83 g/km
```

Pickups and standard SUVs also show elevated emissions.

This suggests that **changing fleet composition away from large vans, SUVs and trucks toward more compact vehicles could have a larger aggregate effect than engine-size regulation alone**.

### 2. Manufacturer / Brand

Exotic and performance-oriented manufacturers show substantial positive CO2 premiums even after controlling for engine size and vehicle class.

However, coefficients for brands with very few observations should be interpreted cautiously.

### 3. Engine Size

Engine size remains a consistent, moderate and directly actionable driver.

The model estimates approximately:

```text
+41 g/km CO2
```

per standard deviation increase in engine size.

---

## 🧪 Residual Analysis

The largest prediction errors revealed two meaningful patterns.

### Hybrid Vehicles

Vehicles such as:

* Hyundai Ioniq
* Kia Niro
* Volkswagen Jetta Turbo Hybrid

were often **over-predicted**.

The reason is that `Fuel Type` does not adequately distinguish hybrid drivetrains. The model therefore lacks an important signal for unusually efficient hybrid vehicles.

### Performance Vehicles

Vehicles such as:

* Ford GT
* Audi R8
* Chevrolet Corvette
* Lamborghini Aventador

were often **under-predicted**.

Their performance-oriented tuning creates higher emissions than expected from engine size, brand and vehicle class alone.

These findings demonstrate why the high-end outliers were retained rather than removed as "bad data."

---

## 🚗 Business Recommendations

### 1. Prioritize Fleet Composition

Shifting the vehicle mix away from:

```text
Large SUVs
Vans
Trucks
```

toward:

```text
Compact / lower-emission vehicle classes
```

is the largest lever identified by the analysis.

### 2. Continue Engine Downsizing

Engine downsizing remains an important complementary strategy.

Its effect is independent enough of vehicle-class effects to justify treating it as a separate policy/design lever.

### 3. Improve Hybrid Representation

Future datasets should explicitly capture:

```text
Hybrid / Plug-in Hybrid / ICE / EV
```

rather than relying solely on broad fuel-type categories.

This would allow the model to distinguish the unusually low emissions of hybrid drivetrains.

### 4. Treat Brand Effects Carefully

Brand-level coefficients can reveal meaningful patterns, but estimates for manufacturers with very few observations should not be treated as robust population-level effects.

---

## 🧠 Key Machine Learning Concepts Demonstrated

This project demonstrates practical application of:

* Exploratory Data Analysis
* Data Cleaning
* Duplicate Detection
* Outlier Analysis
* Correlation Analysis
* Variance Inflation Factor (VIF)
* Multicollinearity Diagnosis
* Feature Engineering
* Feature Selection
* High-Cardinality Categorical Variables
* Target-Encoding Leakage
* One-Hot Encoding
* Standard Scaling
* Train-Test Splitting
* Data Leakage Prevention
* Ridge Regression
* Cross-Validated Hyperparameter Selection
* R²
* RMSE
* MAE
* Residual Analysis
* Model Interpretability
* Business-oriented Feature Selection

---

## 🛠️ Tech Stack

```text
Python
│
├── NumPy
├── Pandas
├── SciPy
├── Matplotlib
├── Seaborn
├── Scikit-learn
└── Statsmodels
```

Key libraries used in the analysis include:

```python
numpy
pandas
scipy
matplotlib
seaborn
scikit-learn
statsmodels
```

---

## 📁 Project Structure

```text
CO2-Emission-Case-Study/
│
├── data/
│   └── CO2_Emissions.csv
│
├── notebooks/
│   └── CO2_Emission_Analysis.ipynb
│
├── README.md
│
└── requirements.txt
```

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd CO2-Emission-Case-Study
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows**

```bash
.venv\Scripts\activate
```

**macOS/Linux**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the analysis notebook and execute the cells sequentially.

---

## 🎯 Final Takeaway

This case study demonstrates that a strong regression project is not simply about obtaining the highest possible predictive score.

The final model deliberately sacrifices some potentially available predictive information in favor of **causal relevance, interpretability, and actionable business insights**.

Using only:

```text
Engine Size
Make
Vehicle Class
Transmission Family
Fuel Type
```

the Ridge model achieves:

```text
R²   ≈ 0.87
MAE  ≈ 16.25 g/km
RMSE ≈ 21.47 g/km
```

while providing interpretable evidence that **vehicle class, manufacturer characteristics, and engine size are important drivers of CO2 emissions**.

The most important practical conclusion is that **fleet composition and vehicle-class choices may provide a larger emissions-reduction opportunity than engine downsizing alone**, with engine downsizing serving as an important complementary lever.

---

## 👤 Author

**Divya Rani**

Data / Machine Learning Engineer

