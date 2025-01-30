import pandas as pd

# File pathT
file_path = r'C:\\Users\\bhavana\\OneDrive\\Desktop\\Aspire Tech\\Companies\\Nuel Inc\\Sample - Superstore.csv'

# Load the dataset
superstore_data = pd.read_csv(file_path, encoding='latin1')

# Step 1: Handle Missing Values
# Check for missing values
print("Missing values:")
print(superstore_data.isnull().sum())

# Step 2: Convert Dates to Proper Format
superstore_data['Order Date'] = pd.to_datetime(superstore_data['Order Date'], format='%m/%d/%Y')
superstore_data['Ship Date'] = pd.to_datetime(superstore_data['Ship Date'], format='%m/%d/%Y')

# Step 3: Remove Duplicates
superstore_data.drop_duplicates(inplace=True)

# Step 4: Add Calculated Fields
# Calculate delivery time in days
superstore_data['Delivery Time (Days)'] = (superstore_data['Ship Date'] - superstore_data['Order Date']).dt.days

# Calculate profit margin
superstore_data['Profit Margin'] = superstore_data['Profit'] / superstore_data['Sales']

# Step 5: Convert Dates Back to Desired Format (MM/DD/YYYY)
superstore_data['Order Date'] = superstore_data['Order Date'].dt.strftime('%m/%d/%Y')
superstore_data['Ship Date'] = superstore_data['Ship Date'].dt.strftime('%m/%d/%Y')

# Save the cleaned dataset
cleaned_file_path = 'Cleaned_Superstore_Data.csv'
superstore_data.to_csv(cleaned_file_path, index=False)

print(f"Cleaned dataset saved to {cleaned_file_path}")

##Exploratory Data Analysis##
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

## Advance Analysis##
import pandas as pd
import statsmodels.api as sm
# Step 1: Load the dataset
file_path = r'C:\Users\bhavana\OneDrive\Desktop\Aspire Tech\Companies\Nuel Inc\Cleaned - Sample - Superstore.csv'
superstore_data = pd.read_csv(file_path, encoding='ISO-8859-1')

# Step 2: Prepare data for regression
regression_data = superstore_data[['Sales', 'Discount', 'Profit', 'Profit Margin']].dropna()

# Step 3: Define dependent and independent variables
X = regression_data[['Discount', 'Profit', 'Profit Margin']]
y = regression_data['Sales']

# Step 4: Add a constant for the intercept
X = sm.add_constant(X)

# Step 5: Build and fit the regression model
model = sm.OLS(y, X).fit()

# Step 6: Display the summary of the regression model
print(model.summary())

 ## Multicollinearity Analysis##
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

# Prepare the data for regression
regression_data = superstore_data[['Sales', 'Discount', 'Profit', 'Profit Margin']].dropna()

# Correlation matrix
correlation_matrix = regression_data[['Discount', 'Profit', 'Profit Margin']].corr()
print("Correlation Matrix:")
print(correlation_matrix)

# VIF calculation
X = regression_data[['Discount', 'Profit', 'Profit Margin']]
X = sm.add_constant(X)  # Add constant for intercept in VIF
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print("\nVariance Inflation Factor (VIF):")
print(vif_data)

##Residual Analysis##

import matplotlib.pyplot as plt
import scipy.stats as stats

# Get residuals from the model
residuals = model.resid

# Plot histogram of residuals
plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=30, color='skyblue', edgecolor='black')
plt.title('Histogram of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.grid()
plt.show()

# QQ plot for residuals
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('QQ Plot of Residuals')
plt.grid()
plt.show()

 ## Add more variables##
import pandas as pd
import statsmodels.api as sm

# File path
file_path = r'C:\Users\bhavana\OneDrive\Desktop\Aspire Tech\Companies\Nuel Inc\Cleaned - Sample - Superstore.csv'

# Load dataset
superstore_data = pd.read_csv(file_path, encoding='ISO-8859-1')

# Define dependent variable (y_new)
y_new = superstore_data['Sales']

# Define independent variables (X_new)
X_new = superstore_data[['Discount', 'Profit', 'Profit Margin']]

# Add dummy variables
regression_data_with_dummies = pd.get_dummies(superstore_data, columns=['Region', 'Category'], drop_first=True)
X_new = regression_data_with_dummies[['Discount', 'Profit', 'Profit Margin', 'Region_South', 'Category_Office Supplies', 'Category_Technology']]

# Convert `bool` columns to `float`
X_new = X_new.astype(float)

# Add constant to X_new
X_new = sm.add_constant(X_new)

# Fit the regression model
try:
    model_with_new_vars = sm.OLS(y_new, X_new).fit()
    print(model_with_new_vars.summary())
except Exception as e:
    print(f"Error fitting the model: {e}")

##Step 2: Customer Segmentation##
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

# Prepare data for clustering
customer_data = superstore_data.groupby('Customer ID').agg({
    'Sales': 'sum',
    'Profit': 'sum',
    'Discount': 'mean'
}).reset_index()
# Use only numeric columns for clustering
clustering_features = customer_data[['Sales', 'Profit', 'Discount']]
# Apply K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
customer_data['Cluster'] = kmeans.fit_predict(clustering_features)
# Visualize the clusters
sns.scatterplot(data=customer_data, x='Sales', y='Profit', hue='Cluster', palette='viridis')
plt.title('Customer Segmentation')
plt.xlabel('Total Sales')
plt.ylabel('Total Profit')
plt.show()

##Step 3 Analyze Low-Performing Products or Categories ##
##Code to Identify Low-Performing Products:##
# Low-performing products
low_performing_products = superstore_data.groupby('Product ID').agg({
    'Sales': 'sum',
    'Profit': 'sum'
}).sort_values(by='Sales').head(10)

print("Low-Performing Products:")
print(low_performing_products)

##Code to Identify Low-Performing Categories:##
# Low-performing categories
low_performing_categories = superstore_data.groupby('Category').agg({
    'Sales': 'sum',
    'Profit': 'sum'
}).sort_values(by='Sales')

print("Low-Performing Categories:")
print(low_performing_categories)

##optimal enhancements##
##correlation analysis## ##correlation matrix and heat map##
import seaborn as sns
import matplotlib.pyplot as plt

# Select only numeric columns for correlation
numeric_data = superstore_data.select_dtypes(include=['float64', 'int64'])

# Check if there are any NaN values and handle them
numeric_data = numeric_data.fillna(0)  # Replace NaN with 0 (or use a strategy like mean imputation)

# Compute the correlation matrix
correlation_matrix = numeric_data.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix of Numerical Columns')
plt.show()

##Geographical Analysis##
import pandas as pd
# Aggregate data by City and State
geo_data = superstore_data.groupby(['City', 'State']).agg({
    'Sales': 'sum',
    'Profit': 'sum'
}).reset_index()

# Identify underperforming locations
underperforming_locations = geo_data[geo_data['Profit'] < 0]
# Display underperforming locations
print("Underperforming Locations:")
print(underperforming_locations)
##Code: Visualize Geographic Analysis with a Bar Plot##
# Sort by Profit and visualize
plt.figure(figsize=(12, 6))
sns.barplot(data=geo_data.sort_values(by='Profit'), x='State', y='Profit', palette='coolwarm')
plt.xticks(rotation=90)
plt.title('Profit by State')
plt.ylabel('Total Profit')
plt.xlabel('State')
plt.show()














