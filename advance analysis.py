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
