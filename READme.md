**Retail Sales Analysis and Insights for RetailPulse**

## Table of Contents

1. [Introduction](#introduction)
2. [Data Overview](#data-overview)
3. [Methodology](#methodology)
    - [3.1. Environment Setup](#31-environment-setup)
    - [3.2. Data Loading](#32-data-loading)
    - [3.3. Data Cleaning and Preprocessing](#33-data-cleaning-and-preprocessing)
    - [3.4. Exploratory Data Analysis (EDA)](#34-exploratory-data-analysis-eda)
    - [3.5. Customer Segmentation](#35-customer-segmentation)
    - [3.6. Regression and Profitability Analysis](#36-regression-and-profitability-analysis)
    - [3.7. Holiday Impact Analysis](#37-holiday-impact-analysis)
    - [3.8. Identifying Low-Performing Products](#38-identifying-low-performing-products)
    - [3.9. Power BI Dashboard Integration](#39-power-bi-dashboard-integration)
4. [Key Performance Indicators (KPIs)](#key-performance-indicators-kpis)
5. [Findings and Recommendations](#findings-and-recommendations)
6. [Challenges & Assumptions](#challenges--assumptions)
7. [Future Enhancements](#future-enhancements)

---

## Introduction

RetailPulse, a **retail analytics company**, aims to leverage data-driven insights to optimize decision-making. This study **analyzes retail sales trends** to identify **high-performing products, profitable customer segments, and seasonal sales variations**. 

By integrating **internal sales data** with **external holiday data**, the study provides actionable insights to **enhance sales performance, improve customer retention, and optimize discounting strategies**.

---

## Data Overview

### Datasets Utilized

1. **Superstore Sales Data (`superstore_sales.csv`):**
    - **Description:** Contains retail transactions, customer information, and product sales.
    - **Key Columns:**
        - `Order ID`: Unique identifier for each order.
        - `Customer ID`: Unique customer identifier.
        - `Category`: Product category (Technology, Furniture, Office Supplies).
        - `Sales`: Revenue generated from the transaction.
        - `Profit`: Net profit/loss for the transaction.
        - `Discount`: Discount percentage applied.

2. **Holiday Data (`holiday.csv`):**
    - **Description:** Federal holiday dataset used to assess holiday sales impact.
    - **Key Columns:**
        - `Date`: Holiday date.
        - `Holiday Name`: Name of the federal holiday.

### Data Scope

- **Time Period:** 2014 - 2018.
- **Volume:** Thousands of transactions across multiple regions.
- **Geographical Coverage:** United States.

---

## 3. Methodology

The analysis follows a structured approach to **handle large datasets efficiently** and extract **meaningful insights**.

### 3.1. Environment Setup
**Libraries Imported:**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from collections import defaultdict, Counter
import gc


3.2. Data Loading
python
Copy
Edit
SALES_FILE = "superstore_sales.csv"
HOLIDAY_FILE = "holiday.csv"

sales_data = pd.read_csv(SALES_FILE)
holiday_data = pd.read_csv(HOLIDAY_FILE)
3.3. Data Cleaning and Preprocessing
Handle Missing Values: Drop rows with missing or invalid entries.
Standardize Date Formats: Convert Order Date and Date columns to datetime format.
Merge Datasets: Combine sales_data and holiday_data based on Order Date.
python
Copy
Edit
sales_data['Order Date'] = pd.to_datetime(sales_data['Order Date'])
holiday_data['Date'] = pd.to_datetime(holiday_data['Date'])

sales_data = sales_data.merge(holiday_data, how="left", left_on="Order Date", right_on="Date")
sales_data["Holiday_Flag"] = sales_data["Date"].notnull().astype(int)
3.4. Exploratory Data Analysis (EDA)
Sales Trends Analysis
python
Copy
Edit
sales_trends = sales_data.groupby("Order Date")["Sales"].sum()

plt.figure(figsize=(12, 6))
plt.plot(sales_trends)
plt.title("Sales Trends Over Time")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.show()
3.5. Customer Segmentation
Clustering Customers Using K-Means
python
Copy
Edit
from sklearn.cluster import KMeans

customer_data = sales_data.groupby("Customer ID").agg({"Sales": "sum", "Profit": "sum", "Discount": "mean"})
kmeans = KMeans(n_clusters=4, random_state=42).fit(customer_data)
customer_data["Segment"] = kmeans.labels_
3.6. Regression and Profitability Analysis
Ordinary Least Squares (OLS) Regression
python
Copy
Edit
X = sales_data[["Discount", "Profit"]]
y = sales_data["Sales"]

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())
3.7. Holiday Impact Analysis
python
Copy
Edit
avg_sales_holidays = sales_data[sales_data["Holiday_Flag"] == 1]["Sales"].mean()
avg_sales_non_holidays = sales_data[sales_data["Holiday_Flag"] == 0]["Sales"].mean()

print(f"Average Sales on Holidays: ${avg_sales_holidays:.2f}")
print(f"Average Sales on Non-Holidays: ${avg_sales_non_holidays:.2f}")
3.8. Identifying Low-Performing Products
python
Copy
Edit
low_perf_products = sales_data.groupby("Product ID").agg({"Sales": "sum", "Profit": "sum"}).sort_values(by="Profit")

plt.figure(figsize=(10, 5))
sns.barplot(x=low_perf_products.index[:10], y=low_perf_products["Profit"][:10], palette="coolwarm")
plt.xticks(rotation=90)
plt.title("Low-Performing Products by Profit")
plt.show()
3.9. Power BI Dashboard Integration
Key Visualizations
Sales Trends: Line Chart with Order Date on X-axis and Total Sales on Y-axis.
Customer Segmentation: Pie Chart of customer spending categories.
Profit by Region: Map with color scale for profit.
Holiday Impact: Bar Chart comparing holiday vs. non-holiday sales.
Steps to Load Data in Power BI
Open Power BI Desktop.
Click Home > Get Data > CSV.
Select cleaned_sales_data.csv.
Click Load to import data.
Key Performance Indicators (KPIs)
Total Sales Growth Rate
Sales by Category and Store
Customer Segmentation Metrics
Holiday Impact on Sales
Profitability Analysis (Revenue - Cost)
Discount vs. Sales Relationship
Regional Performance Metrics
Findings and Recommendations
Key Insights
✅ Technology and Office Supplies dominate sales
✅ Discounts negatively impact profit margins
✅ Holiday promotions increase transactions but lower per-transaction revenue
✅ Certain states (Texas, Pennsylvania) have negative profit margins

Strategic Recommendations
🔹 Optimize holiday promotions to maintain profitability.
🔹 Focus retention efforts on high-value customers.
🔹 Reduce excessive discounting on low-margin products.
🔹 Improve logistics in underperforming regions.

Challenges & Assumptions
Assumption: Federal holidays uniformly impact all regions.
Challenge: High multicollinearity between Discount and Profit Margin.
Future Enhancements
🔹 Incorporate local holidays for better regional analysis.
🔹 Expand into predictive analytics for sales forecasting.
🔹 Improve shipping & logistics data integration.
