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

### 3.2. Environment Setup**
SALES_FILE = "superstore_sales.csv"
HOLIDAY_FILE = "holiday.csv"

sales_data = pd.read_csv(SALES_FILE)
holiday_data = pd.read_csv(HOLIDAY_FILE)

### 3.3. Data Cleaning and Preprocessing
Handle Missing Values: Drop rows with missing or invalid entries.
Standardize Date Formats: Convert Order Date and Date columns to datetime format.
Merge Datasets: Combine sales_data and holiday_data based on Order Date.




