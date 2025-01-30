#Retail Sales Analysis and Insights for RetailPulse
Overview
This project analyzes retail sales data to provide actionable insights for RetailPulse. It integrates internal transaction data with external holiday data, leveraging Power BI dashboards, advanced analytics (regression, clustering), and data-driven recommendations to optimize business decisions.

📖 Table of Contents
Problem Definition
Datasets Used
Methodology
Key Findings
Recommendations
Challenges & Assumptions
Code & Scripts
Power BI Dashboard Integration
Future Enhancements
🔍 Problem Definition
Objective:
RetailPulse seeks to optimize sales, marketing, and pricing strategies by analyzing:

Sales performance by region, product category, and customer segment
Impact of discounts, delivery times, and holidays on revenue
Customer behavior, including spending patterns and repeat purchases
Performance of low-performing products and regions
Data-driven insights for pricing, promotions, and inventory management
📂 Datasets Used
1️⃣ Internal Dataset: Superstore Sales Data
Transaction details (Order ID, Customer ID, Product, Category, Sales, Profit, Discount)
Customer demographics (City, State, Region)
Order details (Order Date, Ship Date, Ship Mode)
2️⃣ External Dataset: Holiday Data
Public holiday dataset (Year, Holiday Name, Date)
Created a holiday.csv file to merge with sales data
⚙️ Methodology
🔹 1. Data Preparation
Cleaning: Handled missing values, duplicates, inconsistent formats
Merging: Combined internal & external datasets on Order Date
Feature Engineering: Created new features like Holiday Flag
🔹 2. Exploratory Data Analysis (EDA)
Sales Trends: Time-series analysis
Customer Segmentation: K-Means clustering on spending habits
Product & Regional Insights: Identified top & low-performing categories
Holiday Impact: Compared holiday vs. non-holiday sales
🔹 3. Advanced Analytics
Regression Analysis: Impact of discounts, profit margins, holidays on sales
Clustering (K-Means): Segmented customers into Low, Medium, High, Very High spenders
Low-Performing Analysis: Identified low-profit regions and products
🔹 4. Data Visualization
Power BI dashboards for interactive reporting
Filters for Region, Category, Customer Segment
Visualizations for Sales trends, Profitability, Customer segmentation, Holiday impact
📊 Key Findings
📈 Sales Insights
✅ Total Sales: $1.8M across 4,407 transactions
✅ Top-selling categories: Technology & Office Supplies
✅ High-revenue cities: New York, Los Angeles, Seattle

🎯 Customer Insights
✅ Top 10% of customers drive ~70% of revenue
✅ Low-spending customers make up 50% of transactions but contribute only 10% of revenue

📆 Holiday Impact
✅ More transactions occur during holidays, but average sales per transaction are lower
✅ Discounts reduce profitability, requiring careful promotion planning

🏙️ Regional Performance
✅ Top states: California, New York, Washington
✅ Underperforming states: Texas, Pennsylvania (negative profits)

✅ Recommendations
🛒 Sales Optimization
🔹 Promote high-margin products during peak sales periods
🔹 Bundle low-performing products with top-selling items

📢 Marketing & Customer Retention
🔹 Loyalty programs & personalized offers for high-spending customers
🔹 Target low & medium spenders with incentives to boost spending

🎁 Holiday-Specific Strategies
🔹 Offer bundled discounts instead of blanket discounts to maintain profit margins

