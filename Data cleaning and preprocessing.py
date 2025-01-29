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













