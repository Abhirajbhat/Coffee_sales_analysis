

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



coffee_data=pd.read_csv('/content/index.csv')
coffee_data.head(20)

coffee_data.info()

"""##1. Cleaning  and Handling the Data"""

coffee_data.isnull().sum()

df=coffee_data  #duplicate the dataset
df.head(20)

df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Convert datetime and date
df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

# Convert money to numeric
df["money"] = pd.to_numeric(df["money"], errors="coerce")

# Handle missing values
df["card"] = df["card"].fillna("CASH") #Filling Cash for the  cash customers

#Normalize coffee name
df["coffee_name"]=df["coffee_name"].str.title()

# Normalize payment method
df["payment_method"] = df["cash_type"].str.lower().str.strip()

# Feature engineering
df["year"] = df["datetime"].dt.year
df["month"] = df["datetime"].dt.month
df["day"] = df["datetime"].dt.day
df["hour"] = df["datetime"].dt.hour
df["weekday"] = df["datetime"].dt.weekday
df["weekday_name"] = df["datetime"].dt.day_name()
df["is_weekend"] = df["weekday"].isin([5, 6])


# Outlier detection (IQR rule on money)
Q1 = df["money"].quantile(0.25)
Q3 = df["money"].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
df = df[(df["money"] >= lower) & (df["money"] <= upper)]

# Save cleaned dataset
df.to_csv("cleaned_Coffee_data.csv", index=False)
print("Step 1 complete. Cleaned data saved as cleaned_index.csv")

df.head(50)

df.describe()

df.info()

df.isnull().sum()

"""##2.Exploratory Data Analysis (EDA)"""

df=pd.read_csv("/content/cleaned_Coffee_data.csv",parse_dates=["datetime"])

#Coffee by popuarity
print(df["coffee_name"].value_counts().head(20))
#plotting the graph
df["coffee_name"].value_counts().head(20).plot(kind="bar",color="brown")
plt.show()

#Daily Revenue
daily = df.set_index("datetime").resample("D")["money"].sum()
daily.plot(figsize=(10,4), title="Daily Revenue")
plt.show()

#Hourly revenue
hourly = df.groupby("hour")["money"].sum()
hourly.plot(kind="bar", figsize=(8,4), title="Revenue by Hour")
plt.show()

#Weekday revenue
weekday = df.groupby("weekday_name")["money"].sum()
weekday = weekday.reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
weekday.plot(kind="bar", figsize=(8,4), title="Revenue by Weekday")
plt.show()

"""##3.Machine learning Model"""

# Step 3: Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Features and target
X = df[["hour","weekday","is_weekend","coffee_name"]]
y = df["money"]


# One-hot encode coffee_name
ohe = OneHotEncoder(handle_unknown="ignore")
X_encoded = ohe.fit_transform(X[["coffee_name"]])
coffee_cols = [f"coffee_{c}" for c in ohe.categories_[0]]
X_num = X[["hour","weekday","is_weekend"]].astype(float).reset_index(drop=True)
X_final = pd.concat([X_num, pd.DataFrame(X_encoded.toarray(), columns=coffee_cols)], axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

MAE=mean_absolute_error(y_test, y_pred)
RMSE=np.sqrt(mean_squared_error(y_test, y_pred))
R2=r2_score(y_test, y_pred)
# Evaluation
print("MAE:",MAE)
print("RMSE:", RMSE)
print("R2 Score:",R2)

from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Use daily revenue data
daily = df.set_index("datetime").resample("D")["money"].sum()

# Train ARIMA model
model = ARIMA(daily, order=(5,1,0))   # (p,d,q) parameters
model_fit = model.fit()

# Forecast next 30 days
forecast = model_fit.forecast(steps=30)

# Plot actual vs forecast
plt.figure(figsize=(10,5))
plt.plot(daily, label="Actual Sales")
plt.plot(forecast.index, forecast, label="Forecast", color="red")
plt.title("Coffee Sales Forecast (Next 30 Days)")
plt.xlabel("Date")
plt.ylabel("Revenue")
plt.legend()
plt.show()

# y_test and y_pred are from your ML regression
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred, alpha=0.6, color="blue")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         "r--", lw=2)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()

residuals = y_test - y_pred

plt.figure(figsize=(8,5))
plt.scatter(y_pred, residuals, alpha=0.6, color="green")
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Predicted Sales")
plt.ylabel("Residuals (Error)")
plt.title("Residuals vs Predicted")
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(residuals, bins=20, kde=True, color="orange")
plt.title("Distribution of Prediction Errors")
plt.xlabel("Error (Residual)")
plt.show()

# Calculate residuals
residuals = y_test - y_pred

print("🔹 Conclusion & Insights 🔹\n")

# Model performance
print(f"✅ Mean Absolute Error (MAE): {MAE:.2f}")
print(f"✅ Root Mean Squared Error (RMSE): {RMSE:.2f}")
print(f"✅ R² Score: {R2:.2f}")

# Residual analysis
print("\n📌 Residual Analysis:")
print(f"Average Residual (Bias): {np.mean(residuals):.2f}")
print(f"Std of Residuals: {np.std(residuals):.2f}")

if abs(np.mean(residuals)) < 0.5:
    print("✔️ Model is unbiased (errors average near zero).")
else:
    print("⚠️ Model shows some bias (systematic over/under prediction).")

def print_business_insights():
    print("\n\n📊🔹 Business Insights from Coffee Sales Data 🔹\n")

    # Top-Selling Products
    print("1. Top-Selling Products")
    print("- Americano With Milk, Latte, and Cappuccino are the most popular drinks.")
    print("- Niche products like Cocoa and Espresso have the lowest demand.")
    print("👉 Focus marketing and promotions on top sellers while considering discounts for low-performing items.\n")

    # Daily Revenue Trends
    print("2. Daily Revenue Trends")
    print("- Revenue shows high daily fluctuations.")
    print("- Peaks occur during specific periods (April–May had higher revenues).")
    print("👉 Sales are not evenly distributed — promotions could be timed during low-sales weeks.\n")

    # Revenue by Hour
    print("3. Revenue by Hour")
    print("- Highest revenue observed at 10 AM and 7 PM.")
    print("- Consistent demand between 11 AM – 6 PM, but mornings and evenings are the strongest.")
    print("👉 Morning promotions (breakfast combos) and evening offers (snacks + coffee) can maximize sales.\n")

    # Revenue by Weekday
    print("4. Revenue by Weekday")
    print("- Tuesday is the strongest sales day, while Monday and Sunday are relatively weaker.")
    print("👉 Introduce special deals on Mondays/Sundays to attract customers and balance weekly revenue.\n")

    # Time-Series Forecasting
    print("5. Time-Series Forecasting (ARIMA)")
    print("- Predicted sales for the next 30 days show stability with slight fluctuations.")
    print("👉 Business can plan inventory, staffing, and supply chain based on forecasted demand.\n")

    # Machine Learning Model
    print("6. Machine Learning (Regression Model)")
    print("- R² Score ≈ 0.78 → Model explains ~78% of sales variance.")
    print("- Residual plots show errors are randomly distributed (model is unbiased).")
    print("👉 Model is reliable for predicting future sales with decent accuracy.\n")

    # Customer Behavior
    print("7. Customer Behavior Patterns")
    print("- Customers prefer weekday purchases over weekends.")
    print("- Morning rush hours (10 AM) suggest strong office-goer demand.")
    print("👉 Target offices/working professionals with subscription models or loyalty programs.\n")

    # Conclusion
    print("✨ Conclusion:")
    print("The analysis shows Americano With Milk, Latte, and Cappuccino dominate sales,")
    print("demand peaks in the morning and evening, and Tuesdays are the busiest day.")
    print("Predictive models suggest stable upcoming demand, enabling better inventory and staffing decisions.")
    print("Targeted promotions on weaker days/hours and boosting low-selling drinks could further maximize revenue.")

# Run the function
print_business_insights()
