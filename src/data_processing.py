import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.df_clean = None
        self.customer_features = None
        
    def load_data(self):
        """Load data from CSV file"""
        try:
            self.df = pd.read_csv(self.file_path, encoding='ISO-8859-1')
            print("Data loaded successfully!")
            return self.df
        except Exception as e:
            print(f"Failed to load data: {str(e)}")
            return None

    def clean_data(self):
        """Clean and preprocess data with enhanced features"""
        if self.df is None:
            print("Please load data first!")
            return None
            
        self.df_clean = self.df.copy()
        print("Starting enhanced data cleaning process...")
        
        # 1. Basic Data Cleaning
        initial_rows = len(self.df_clean)
        self.df_clean = self.df_clean.dropna()
        print(f"Removed missing values: {initial_rows - len(self.df_clean)} rows")
        
        # 2. Remove duplicates
        before_duplicates = len(self.df_clean)
        self.df_clean = self.df_clean.drop_duplicates()
        print(f"Removed duplicate entries: {before_duplicates - len(self.df_clean)} rows")
        
        # 3. Data type conversion and standardization
        self.df_clean['InvoiceNo'] = self.df_clean['InvoiceNo'].astype(str)
        self.df_clean['StockCode'] = self.df_clean['StockCode'].astype(str)
        self.df_clean['CustomerID'] = pd.to_numeric(self.df_clean['CustomerID'], errors='coerce')
        
        # 4. Handle cancelled orders and validate invoice numbers
        orders_before = len(self.df_clean)
        self.df_clean = self.df_clean[~self.df_clean['InvoiceNo'].str.startswith('C')]
        print(f"Removed cancelled orders: {orders_before - len(self.df_clean)} rows")
        
        # 5. Convert and enrich datetime information
        self.df_clean['InvoiceDate'] = pd.to_datetime(self.df_clean['InvoiceDate'], format='%m/%d/%Y %H:%M')
        self.df_clean['Year'] = self.df_clean['InvoiceDate'].dt.year
        self.df_clean['Month'] = self.df_clean['InvoiceDate'].dt.month
        self.df_clean['Day'] = self.df_clean['InvoiceDate'].dt.day
        self.df_clean['Hour'] = self.df_clean['InvoiceDate'].dt.hour
        self.df_clean['DayOfWeek'] = self.df_clean['InvoiceDate'].dt.dayofweek
        self.df_clean['Weekend'] = self.df_clean['DayOfWeek'].isin([5, 6]).astype(int)
        
        # 6. Handle invalid quantities and prices
        quantity_before = len(self.df_clean)
        self.df_clean = self.df_clean[self.df_clean['Quantity'] > 0]
        self.df_clean = self.df_clean[self.df_clean['UnitPrice'] > 0]
        print(f"Removed invalid quantities/prices: {quantity_before - len(self.df_clean)} rows")
        
        # 7. Calculate monetary values
        self.df_clean['TotalAmount'] = self.df_clean['Quantity'] * self.df_clean['UnitPrice']
        
        # 8. Handle outliers using IQR method
        for column in ['Quantity', 'UnitPrice', 'TotalAmount']:
            Q1 = self.df_clean[column].quantile(0.25)
            Q3 = self.df_clean[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers_mask = (self.df_clean[column] >= lower_bound) & (self.df_clean[column] <= upper_bound)
            outliers_count = (~outliers_mask).sum()
            self.df_clean = self.df_clean[outliers_mask]
            print(f"Removed {outliers_count} outliers from {column}")
        
        # 9. Standardize country names
        self.df_clean['Country'] = self.df_clean['Country'].str.strip().str.upper()
        
        # 10. Create product categories based on description (simple version)
        def categorize_product(description):
            description = str(description).upper()
            if 'GIFT' in description:
                return 'Gifts'
            elif any(word in description for word in ['SET', 'PACK']):
                return 'Sets'
            elif any(word in description for word in ['BAG', 'HANDBAG']):
                return 'Bags'
            elif any(word in description for word in ['CHRISTMAS', 'XMAS']):
                return 'Seasonal'
            else:
                return 'Others'
                
        self.df_clean['ProductCategory'] = self.df_clean['Description'].apply(categorize_product)
        
        # 11. Add transaction metrics
        self.df_clean['ItemsPerTransaction'] = self.df_clean.groupby('InvoiceNo')['Quantity'].transform('sum')
        self.df_clean['UniqueItemsPerTransaction'] = self.df_clean.groupby('InvoiceNo')['StockCode'].transform('nunique')
        
        # Ensure CustomerID is string type for consistency
        self.df_clean['CustomerID'] = self.df_clean['CustomerID'].astype(str)
        
        # Reset index for clean data
        self.df_clean.reset_index(drop=True, inplace=True)
        
        print("\nData cleaning and feature engineering completed!")
        print(f"Final dataset shape: {self.df_clean.shape}")
        
        # Print summary of new features
        print("\nNew features added:")
        print("- Time-based features: InvoiceDay, Year, Month, Day, Hour, DayOfWeek, Weekend")
        print("- Product features: ProductCategory")
        print("- Transaction features: ItemsPerTransaction, UniqueItemsPerTransaction")
        
        return self.df_clean

    def compute_customer_features(self):
        if self.df_clean is None:
            print("Please run clean_data() first!")
            return None
        
        # Total Transactions
        total_transactions = self.df_clean.groupby('CustomerID').size().reset_index(name='Total_Transactions')
        
        # Total Spend per Customer
        total_spend = self.df_clean.groupby('CustomerID')['TotalAmount'].sum().reset_index()
        
        # Last Purchase Date
        last_purchase = self.df_clean.groupby('CustomerID')['InvoiceDate'].max().reset_index()
        last_purchase['LastPurchaseDate'] = last_purchase['InvoiceDate'].dt.date
        last_purchase = last_purchase.drop('InvoiceDate', axis=1)
        
        # Unique Products Purchased
        unique_products = self.df_clean.groupby('CustomerID')['StockCode'].nunique().reset_index(name='Unique_Products_Purchased')
        
        # Favorite Day of Week
        favorite_day = self.df_clean.groupby(['CustomerID', 'DayOfWeek']).size().reset_index(name='Count')
        favorite_day = favorite_day.loc[favorite_day.groupby('CustomerID')['Count'].idxmax()][['CustomerID', 'DayOfWeek']]
        
        # Favorite Hour
        favorite_hour = self.df_clean.groupby(['CustomerID', 'Hour']).size().reset_index(name='Count')
        favorite_hour = favorite_hour.loc[favorite_hour.groupby('CustomerID')['Count'].idxmax()][['CustomerID', 'Hour']]
        
        # Merge customer features
        self.customer_features = total_transactions.merge(total_spend, on='CustomerID')
        self.customer_features = self.customer_features.merge(last_purchase, on='CustomerID')
        self.customer_features = self.customer_features.merge(unique_products, on='CustomerID')
        self.customer_features = self.customer_features.merge(favorite_day, on='CustomerID', how='left')
        self.customer_features = self.customer_features.merge(favorite_hour, on='CustomerID', how='left')
        
        self.customer_features.fillna(0, inplace=True)
        return self.customer_features

if __name__ == "__main__":
    print("Data Processor Ready!")
