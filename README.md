# Housing_price_prediction_Linear_Regression_Random_Forest_Regressor


The objective of this project is to predict house prices using property characteristics such as:

- Area
- Number of bedrooms
- Number of bathrooms
- Number of stories
- Main-road accessibility
- Guestroom availability
- Basement availability
- Hot-water heating
- Air conditioning
- Parking spaces
- Preferred-area status
- Furnishing status

Two regression models are evaluated:

1. **Linear Regression**
2. **Random Forest Regressor**

The notebook uses a logarithmic transformation of the target price to reduce skewness before model training.

The notebook performs several EDA steps:

- Inspects dataset structure using `df.info()`
- Separates numerical and categorical variables
- Examines distributions of features
- Visualizes relationships between features and price
- Checks skewness and kurtosis
- Applies logarithmic transformations to highly skewed variables

  ### Transformation of "area" columns

The original `area` feature had:

- Skewness: **1.321**
- Kurtosis: **2.751**

The notebook applies:

```python
df["area_log"] = np.log1p(df["area"])
```

After transformation:

- Skewness: **0.134**
- Kurtosis: **-0.214**

This makes the distribution considerably closer to a symmetric/normal distribution.

### Transformation of "price" column

The original price distribution had:

- Skewness: **1.212**
- Kurtosis: **1.960**

The notebook applies:

```python
df_encoded["price_log"] = np.log1p(df_encoded["price"])
```

After transformation:

- Skewness: **0.141**
- Kurtosis: **-0.137**


### Binary Encoding

The following categorical variables are converted from `yes/no` to `1/0`:

```text
mainroad
guestroom
basement
hotwaterheating
airconditioning
prefarea
```

## Modeling

### 1. Linear Regression

A standard `LinearRegression` model from Scikit-learn is trained after an **80/20 train-test split**.

```python
model = LinearRegression()
model.fit(x_train, y_train)
```

### 2. Random Forest Regressor

A Random Forest model is trained with:

- `n_estimators = 100`
- `random_state = 42`

```python
rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf.fit(x_train, y_train)
```

---

## Model Evaluation

The models are evaluated using:

- **R² Score**
- **RMSE**
- **MAE**
- **MSE**


### Results Reported in the Notebook
<img width="1030" height="187" alt="image" src="https://github.com/user-attachments/assets/66c279af-5869-49b7-9042-854a04e89f5d" />



- 
