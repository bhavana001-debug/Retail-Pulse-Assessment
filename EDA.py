##Step 1:Load and Review the Data##
import pandas as pd
# File paths
internal_data = r'C:\\Users\\bhavana\\OneDrive\\Desktop\\Aspire Tech\\Companies\\Nuel Inc\\Cleaned - Sample - Superstore.csv'
external_data = r'C:\\Users\\bhavana\\OneDrive\\Desktop\\Aspire Tech\\Companies\\Nuel Inc\\us_federal_holidays_2014_2018.csv'

# Load datasets with specified encoding
superstore_data = pd.read_csv(internal_data, encoding='ISO-8859-1')
holidays_data = pd.read_csv(external_data, encoding='ISO-8859-1')

# Preview data
print("Superstore Data Preview:")
print(superstore_data.head())

print("\nHolidays Data Preview:")
print(holidays_data.head())
# Check for missing values
print("\nMissing Values in Superstore Data:")
print(superstore_data.isnull().sum())

print("\nMissing Values in Holidays Data:")
print(holidays_data.isnull().sum())

##Step 2: Key Metrics and KPIs##

# Check for zero or null values in Profit Margin to avoid division errors
superstore_data = superstore_data[superstore_data['Profit Margin'] > 0]

# Calculate Total Sales
superstore_data['Total_Sales'] = superstore_data['Profit'] / superstore_data['Profit Margin']

# Aggregate KPIs
total_sales = superstore_data['Total_Sales'].sum()
total_transactions = superstore_data['Order ID'].nunique()
average_sales_per_transaction = total_sales / total_transactions

print(f"Total Sales: ${total_sales:,.2f}")
print(f"Total Transactions: {total_transactions}")
print(f"Average Sales per Transaction: ${average_sales_per_transaction:,.2f}")
#Step3 ##
import matplotlib.pyplot as plt
# Convert 'Order Date' to datetime format
superstore_data['Order Date'] = pd.to_datetime(superstore_data['Order Date'], errors='coerce')
# Check for any invalid dates
print("Missing or invalid dates in 'Order Date':")
print(superstore_data['Order Date'].isnull().sum())
# Drop rows with invalid dates, if necessary
superstore_data = superstore_data.dropna(subset=['Order Date'])
# Group sales by 'Order Date'
daily_sales = superstore_data.groupby('Order Date')['Sales'].sum().reset_index()
# Plot sales trends
plt.figure(figsize=(12, 6))
plt.plot(daily_sales['Order Date'], daily_sales['Sales'], label='Daily Sales', marker='o')
plt.title('Daily Sales Trends')
plt.xlabel('Order Date')
plt.ylabel('Sales')
plt.xticks(rotation=45)
plt.legend()
plt.grid()
plt.show()

#step4##
import matplotlib.pyplot as plt
# Group spending behavior by customer
customer_spending = superstore_data.groupby('Customer ID')['Sales'].sum().reset_index()
# Segment customers into quartiles based on spending
customer_spending['Segment'] = pd.qcut(customer_spending['Sales'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
# Display segmented customer data
print("Customer Segmentation Data:")
print(customer_spending.head())
# Plot customer segments
plt.figure(figsize=(8, 6))
customer_spending['Segment'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Customer Segmentation')
plt.xlabel('Segments')
plt.ylabel('Number of Customers')
plt.grid()
plt.show()
##
# Calculate average spending for each segment
average_spending = customer_spending.groupby('Segment')['Sales'].mean()
print("Average Spending in Each Segment:")
print(average_spending)
##
# Bar chart for average spending by segment
plt.figure(figsize=(8, 6))
average_spending.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Average Spending by Segment')
plt.xlabel('Segment')
plt.ylabel('Average Spending ($)')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
##
# Calculate total revenue and percentage contribution for each segment
segment_revenue = customer_spending.groupby('Segment')['Sales'].sum()
total_revenue = segment_revenue.sum()
segment_percentage = (segment_revenue / total_revenue) * 100
# Combine results into a DataFrame
revenue_analysis = pd.DataFrame({
    'Total Revenue': segment_revenue,
    'Percentage Contribution': segment_percentage
})
print("Revenue Contribution by Segment:")
print(revenue_analysis)
##
# Pie chart for revenue contribution by segment
plt.figure(figsize=(8, 8))
revenue_analysis['Total Revenue'].plot.pie(
    autopct='%1.1f%%',
    startangle=90,
    colors=['lightcoral', 'gold', 'skyblue', 'lightgreen'],
    labels=revenue_analysis.index,
    wedgeprops={'edgecolor': 'black'}
)
plt.title('Revenue Contribution by Segment')
plt.ylabel('')  # Hide y-label
plt.show()

##
# Filter top customers in the "Very High" segment
top_customers = customer_spending[customer_spending['Segment'] == 'Very High'].sort_values(by='Sales', ascending=False).head(10)
print("Top Customers in 'Very High' Segment:")
print(top_customers)
##
# Bar chart for top customers in the "Very High" segment
plt.figure(figsize=(10, 6))
top_customers.plot(
    x='Customer ID',
    y='Sales',
    kind='bar',
    color='lightblue',
    edgecolor='black'
)
plt.title('Top 10 Customers in Very High Segment')
plt.xlabel('Customer ID')
plt.ylabel('Total Sales ($)')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

##Step5##
# Top-performing products by sales
top_products = superstore_data.groupby('Product ID')['Sales'].sum().nlargest(10).reset_index()
print("Top 10 Products by Sales:")
print(top_products)

# Top-performing cities by sales
top_cities = superstore_data.groupby('City')['Sales'].sum().nlargest(10).reset_index()
print("\nTop 10 Cities by Sales:")
print(top_cities)

##Step 6##
# Convert 'Date' in holidays data to datetime with the correct format
holidays_data['Date'] = pd.to_datetime(holidays_data['Date'], format='%d-%m-%Y', errors='coerce')

# Convert 'Order Date' in superstore data to datetime
superstore_data['Order Date'] = pd.to_datetime(superstore_data['Order Date'], errors='coerce')

# Merge holiday data with sales data
sales_with_holidays = superstore_data.merge(holidays_data, how='left', left_on='Order Date', right_on='Date')

# Check sales on holidays
holiday_sales = sales_with_holidays[sales_with_holidays['Date'].notnull()]
non_holiday_sales = sales_with_holidays[sales_with_holidays['Date'].isnull()]

# Calculate average sales
avg_holiday_sales = holiday_sales['Sales'].mean()
avg_non_holiday_sales = non_holiday_sales['Sales'].mean()

# Print results
print(f"Average Sales on Holidays: ${avg_holiday_sales:,.2f}")
print(f"Average Sales on Non-Holidays: ${avg_non_holiday_sales:,.2f}")

##
##Step7##
# Bar plot for holiday impact
import seaborn as sns

sales_comparison = pd.DataFrame({
    'Type': ['Holiday', 'Non-Holiday'],
    'Average_Sales': [avg_holiday_sales, avg_non_holiday_sales]
})

plt.figure(figsize=(8, 6))
sns.barplot(data=sales_comparison, x='Type', y='Average_Sales', palette='viridis')
plt.title('Holiday vs. Non-Holiday Sales')
plt.xlabel('Type')
plt.ylabel('Average Sales')
plt.grid()
plt.show()
##
import statsmodels.api as sm
# Prepare data for regression
regression_data = superstore_data[['Sales', 'Discount', 'Profit', 'Profit Margin']].dropna()
# Define dependent and independent variables
X = regression_data[['Discount', 'Profit', 'Profit Margin']]  # Independent variables
y = regression_data['Sales']  # Dependent variable

# Add a constant for the intercept
X = sm.add_constant(X)

# Build and fit the regression model
model = sm.OLS(y, X).fit()

# Summary of the regression model
print(model.summary())

