import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import json
from typing import Dict, List, Optional, Union
import os
warnings.filterwarnings('ignore')

class RFMAnalyzer:
    
    def __init__(self, df_clean: pd.DataFrame):
        """
        Initialize RFM Analyzer
        
        Parameters:
        -----------
        df_clean : pd.DataFrame
            Clean transaction-level data with required columns:
            - CustomerID
            - InvoiceDate
            - TotalAmount
            - StockCode
            - Country
            - Hour
            - DayOfWeek
        """
        self._validate_input_data(df_clean)
        self.df_clean = df_clean.copy()
        self.customer_features = None
        self.rfm_results = None
        
        # Define expected data types for validation
        self.expected_types = {
            'CustomerID': 'object',  # should be string
            'Recency': 'int64',
            'Frequency': 'int64',
            'Monetary': 'float64',
            'R_score': 'int64',
            'F_score': 'int64',
            'M_score': 'int64',
            'RFM_Score': 'int64',
            'LastPurchaseDate': 'object',  # should be date string
            'FirstPurchaseDate': 'object',  # should be date string
            'Customer_Lifetime_Days': 'int64',
            'Avg_Transaction_Value': 'float64',
            'Country': 'object'  # should be string
        }
        
        # Define required columns for output
        self.required_columns = [
            'CustomerID',
            'Recency', 'Frequency', 'Monetary',
            'R_score', 'F_score', 'M_score', 'RFM_Score',
            'LastPurchaseDate', 'FirstPurchaseDate',
            'Customer_Lifetime_Days',
            'Avg_Transaction_Value',
            'Country'
        ]
        
    def _validate_input_data(self, df: pd.DataFrame) -> None:
        """Validate input dataframe has required columns"""
        required_columns = [
            'CustomerID', 'InvoiceDate', 'TotalAmount', 
            'StockCode', 'Country', 'Hour', 'DayOfWeek'
        ]
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
    def _validate_output_data(self) -> Dict:
        """Validate output data before saving"""
        validation_results = {}
        
        # Check required columns
        missing_columns = [col for col in self.required_columns if col not in self.rfm_results.columns]
        validation_results['missing_columns'] = missing_columns
        
        # Check data types
        type_issues = {}
        for col in self.rfm_results.columns:
            if col in self.expected_types:
                actual_type = str(self.rfm_results[col].dtype)
                if actual_type != self.expected_types[col]:
                    type_issues[col] = {
                        'expected': self.expected_types[col],
                        'actual': actual_type
                    }
        validation_results['type_issues'] = type_issues
        
        # Check null values
        null_counts = self.rfm_results.isnull().sum().to_dict()
        validation_results['null_counts'] = null_counts
        
        return validation_results
        
    def _format_output_data(self) -> pd.DataFrame:
        """Format data for output"""
        if self.rfm_results is None:
            return None
            
        formatted_df = self.rfm_results.copy()
        
        # Format numeric columns
        numeric_columns = {
            'Recency': '.0f',       # Integer days
            'Frequency': '.0f',     # Integer count
            'Monetary': '.2f',      # 2 decimal places for currency
            'R_score': '.0f',       # Integer score
            'F_score': '.0f',       # Integer score
            'M_score': '.0f',       # Integer score
            'RFM_Score': '.0f',     # Integer score
            'Customer_Lifetime_Days': '.0f',  # Integer days
            'Avg_Transaction_Value': '.2f',   # 2 decimal places for currency
            'Total_Items': '.0f',    # Integer count
            'Unique_Products': '.0f'  # Integer count
        }
        
        # Apply numeric formatting
        for col, format_str in numeric_columns.items():
            if col in formatted_df.columns:
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: f"{float(x):{format_str}}" if pd.notnull(x) else ""
                )
        
        # Ensure CustomerID is string
        if 'CustomerID' in formatted_df.columns:
            formatted_df['CustomerID'] = formatted_df['CustomerID'].astype(str)
        
        # Ensure Country is string
        if 'Country' in formatted_df.columns:
            formatted_df['Country'] = formatted_df['Country'].astype(str)
        
        return formatted_df
        
    def save_results(self, output_dir: str = 'data') -> None:
        """
        Save analysis results with validation and proper formatting.
        Only keeps the latest version of files.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save output files
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Clean up old files
            old_files = [
                'customer_segments.csv',
                'customer_segments_fixed.csv',
                'rfm_results.csv',
                'dashboard_data.json',
                'data_quality_report.json'
            ]
            
            for old_file in old_files:
                try:
                    old_path = os.path.join(output_dir, old_file)
                    if os.path.exists(old_path):
                        os.remove(old_path)
                        print(f"Removed old file: {old_path}")
                except Exception as e:
                    print(f"Warning: Could not remove old file {old_file}: {str(e)}")
            
            # Validate data
            validation_results = self._validate_output_data()
            
            # Log validation results
            print("\nValidation Results:")
            if validation_results['missing_columns']:
                print(f"Missing columns: {validation_results['missing_columns']}")
            if validation_results['type_issues']:
                print("\nData type issues:")
                for col, issue in validation_results['type_issues'].items():
                    print(f"{col}: Expected {issue['expected']}, got {issue['actual']}")
            if any(validation_results['null_counts'].values()):
                print("\nNull value counts:")
                for col, count in validation_results['null_counts'].items():
                    if count > 0:
                        print(f"{col}: {count} nulls")
            
            # Format data
            formatted_results = self._format_output_data()
            
            if formatted_results is not None:
                # Save RFM results
                rfm_path = os.path.join(output_dir, 'customer_segments.csv')
                formatted_results.to_csv(rfm_path, index=False)
                print(f"\nLatest results saved to: {rfm_path}")
                
                # Save dashboard data
                dashboard_data = self.prepare_dashboard_data()
                if dashboard_data is not None:
                    dashboard_path = os.path.join(output_dir, 'dashboard_data.json')
                    with open(dashboard_path, 'w') as f:
                        json.dump(dashboard_data, f, default=str)
                    print(f"Dashboard data saved to: {dashboard_path}")
                
                # Save data quality report
                report_path = os.path.join(output_dir, 'data_quality_report.json')
                with open(report_path, 'w') as f:
                    json.dump({
                        'validation_results': validation_results,
                        'summary_stats': self.get_segment_summary()
                    }, f, indent=2, default=str)
                print(f"Data quality report saved to: {report_path}")
                
        except Exception as e:
            print(f"Error saving results: {str(e)}")
            # Try alternative location
            try:
                alt_dir = os.getcwd()
                rfm_path = os.path.join(alt_dir, 'customer_segments.csv')
                formatted_results.to_csv(rfm_path, index=False)
                print(f"Results saved to alternative location: {rfm_path}")
            except Exception as e:
                print(f"Failed to save to alternative location: {str(e)}")
        
    def prepare_customer_features(self) -> pd.DataFrame:
        """
        Prepare customer level features including main country
        
        Returns:
        --------
        pd.DataFrame
            Customer-level aggregated features
        """
        # Step 1: Aggregate transaction-level features
        transaction_features = self.df_clean.groupby('CustomerID').agg({
            'InvoiceNo': 'nunique',             # Total Transactions
            'TotalAmount': 'sum',               # Total Monetary Value
            'Quantity': 'sum',                  # Total Items
            'StockCode': 'nunique',             # Unique Products
            'Country': lambda x: x.mode().iloc[0] if not x.empty else 'Unknown'  # Most frequent country
        }).reset_index()
        
        # Rename columns
        transaction_features.columns = [
            'CustomerID', 
            'Total_Transactions', 
            'Total_Amount', 
            'Total_Items', 
            'Unique_Products',
            'Country'
        ]
        
        # Calculate Average Transaction Value
        transaction_features['Avg_Transaction_Value'] = (
            transaction_features['Total_Amount'] / 
            transaction_features['Total_Transactions']
        )
        
        # Step 2: Prepare purchase date features
        last_purchase_df = self.df_clean.groupby('CustomerID')['InvoiceDate'].max().reset_index()
        last_purchase_df.columns = ['CustomerID', 'LastPurchaseDate']
        
        first_purchase_df = self.df_clean.groupby('CustomerID')['InvoiceDate'].min().reset_index()
        first_purchase_df.columns = ['CustomerID', 'FirstPurchaseDate']
        
        # Step 3: Merge all features
        customer_features = transaction_features.copy()
        
        # Merge last purchase date
        customer_features = customer_features.merge(
            last_purchase_df, 
            on='CustomerID', 
            how='left'
        )
        
        # Merge first purchase date
        customer_features = customer_features.merge(
            first_purchase_df, 
            on='CustomerID', 
            how='left'
        )
        
        # Calculate Customer Lifetime
        customer_features['Customer_Lifetime_Days'] = (
            customer_features['LastPurchaseDate'] - 
            customer_features['FirstPurchaseDate']
        ).dt.days
        
        # Store and return the prepared features
        self.customer_features = customer_features
        return self.customer_features
        
    def calculate_rfm_scores(self) -> pd.DataFrame:
        """
        Calculate Recency, Frequency, Monetary (RFM) scores
        
        Returns:
        --------
        pd.DataFrame
            RFM scores and metrics for each customer
        """
        if self.customer_features is None:
            self.prepare_customer_features()
            
        self.rfm_results = self.customer_features.copy()
        
        # Calculate Recency
        today = pd.to_datetime('today')
        self.rfm_results['Recency'] = (
            today - self.rfm_results['LastPurchaseDate']
        ).dt.days
        
        # Prepare RFM metrics
        self.rfm_results.rename(columns={
            'Total_Transactions': 'Frequency',
            'Total_Amount': 'Monetary'
        }, inplace=True)
        
        # Calculate RFM scores
        for metric in ['Recency', 'Frequency', 'Monetary']:
            col_name = f'{metric[0]}_score'
            self.rfm_results[col_name] = self._calculate_score(
                self.rfm_results[metric],
                reverse=(metric == 'Recency')
            )
        
        # Calculate RFM Score
        self.rfm_results['RFM_Score'] = (
            self.rfm_results['R_score'] * 100 + 
            self.rfm_results['F_score'] * 10 + 
            self.rfm_results['M_score']
        )
        
        # Convert dates to string format for better serialization
        self.rfm_results['LastPurchaseDate'] = self.rfm_results['LastPurchaseDate'].dt.strftime('%Y-%m-%d')
        self.rfm_results['FirstPurchaseDate'] = self.rfm_results['FirstPurchaseDate'].dt.strftime('%Y-%m-%d')
        
        # Select and return relevant columns including country information
        result_columns = [
            'CustomerID', 'Recency', 'Frequency', 'Monetary',
            'R_score', 'F_score', 'M_score', 'RFM_Score',
            'LastPurchaseDate', 'FirstPurchaseDate', 'Customer_Lifetime_Days',
            'Avg_Transaction_Value', 'Country'
        ]
        
        # Only include columns that exist
        available_columns = [col for col in result_columns if col in self.rfm_results.columns]
        return self.rfm_results[available_columns]
    
    def _calculate_score(self, series: pd.Series, reverse: bool = False) -> pd.Series:
        """Calculate quartile scores for a series"""
        try:
            labels = [4, 3, 2, 1] if not reverse else [1, 2, 3, 4]
            return pd.qcut(series, q=4, labels=labels, duplicates='drop').astype(int)
        except ValueError:
            return pd.cut(series, bins=4, labels=labels).astype(int)
    
    def categorize_customers(self) -> pd.DataFrame:
        """
        Return RFM metrics for visualization
        
        Returns:
        --------
        pd.DataFrame
            Customer RFM metrics for visualization
        """
        if self.rfm_results is None:
            self.calculate_rfm_scores()
            
        return self.rfm_results[['CustomerID', 'Recency', 'Frequency', 'Monetary']]
    
    def prepare_dashboard_data(self) -> Dict:
        """
        Prepare all data needed for dashboard visualization
        
        Returns:
        --------
        Dict
            Dashboard data including sales summary, geographic data,
            time-based analysis, and RFM metrics
        """
        try:
            if self.rfm_results is None:
                self.calculate_rfm_scores()
                
            dashboard_data = {
                'sales_summary': self._prepare_sales_summary(),
                'country_data': self._prepare_geographic_data(),
                'time_analysis': self._prepare_time_analysis(),
                'rfm_data': self._prepare_rfm_data()
            }
            
            return dashboard_data
            
        except Exception as e:
            print(f"Error preparing dashboard data: {str(e)}")
            return None
            
    def _prepare_sales_summary(self) -> Dict:
        """Prepare sales summary metrics"""
        try:
            return {
                'total_customers': int(self.df_clean['CustomerID'].nunique()),
                'total_products': int(self.df_clean['StockCode'].nunique()),
                'total_transactions': int(self.df_clean['InvoiceNo'].nunique()),
                'total_sales': float(self.df_clean['TotalAmount'].sum()),
                'average_order_value': float(self.df_clean.groupby('InvoiceNo')['TotalAmount'].mean().mean())
            }
        except Exception as e:
            print(f"Error preparing sales summary: {str(e)}")
            return {}
    
    def _prepare_geographic_data(self) -> List[Dict]:
        """Prepare geographic analysis data"""
        try:
            geo_data = self.df_clean.groupby('Country').agg({
                'InvoiceNo': 'count',
                'CustomerID': 'nunique',
                'TotalAmount': 'sum'
            }).reset_index()
            
            # Rename columns for clarity
            geo_data.columns = ['country', 'transaction_count', 'customer_count', 'total_sales']
            return geo_data.to_dict('records')
            
        except Exception as e:
            print(f"Error preparing geographic data: {str(e)}")
            return []
    
    def _prepare_time_analysis(self) -> Dict:
        """Prepare time-based analysis data"""
        try:
            # Select only needed columns to avoid conflicts
            time_df = self.df_clean[['InvoiceDate', 'InvoiceNo', 'TotalAmount', 'CustomerID', 'DayOfWeek', 'Hour']].copy()
            
            # Monthly analysis using existing datetime column
            monthly = time_df.groupby([
                'InvoiceDate'
            ]).agg({
                'InvoiceNo': 'count',
                'TotalAmount': 'sum',
                'CustomerID': 'nunique'
            }).reset_index()
            
            # Extract year and month after aggregation
            monthly['Year'] = monthly['InvoiceDate'].dt.year
            monthly['Month'] = monthly['InvoiceDate'].dt.month
            monthly = monthly.groupby(['Year', 'Month']).agg({
                'InvoiceNo': 'sum',
                'TotalAmount': 'sum',
                'CustomerID': 'sum'
            }).reset_index()
            monthly.columns = ['Year', 'Month', 'Number_of_Records', 'Sales', 'Unique_Customers']

            # Daily analysis (by weekday)
            daily = time_df.groupby('DayOfWeek').agg({
                'InvoiceNo': 'count',
                'TotalAmount': 'sum',
                'CustomerID': 'nunique'
            }).reset_index()
            daily['DayName'] = daily['DayOfWeek'].map({
                0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 
                4: 'Fri', 5: 'Sat', 6: 'Sun'
            })
            daily.columns = ['DayOfWeek', 'Number_of_Records', 'Sales', 'Unique_Customers', 'DayName']

            # Hourly analysis
            hourly = time_df.groupby('Hour').agg({
                'InvoiceNo': 'count',
                'TotalAmount': 'sum',
                'CustomerID': 'nunique'
            }).reset_index()
            hourly.columns = ['Hour', 'Number_of_Records', 'Sales', 'Unique_Customers']

            return {
                'monthly': monthly.to_dict('records'),
                'daily': daily.to_dict('records'),
                'hourly': hourly.to_dict('records')
            }

        except Exception as e:
            print(f"Error preparing time analysis: {str(e)}")
            return {
                'monthly': [],
                'daily': [],
                'hourly': []
            }

    
    def _prepare_rfm_data(self) -> List[Dict]:
        """Prepare RFM analysis data"""
        columns = [
            'CustomerID', 'Recency', 'Frequency', 'Monetary',
            'R_score', 'F_score', 'M_score'
        ]
        # Only include columns that exist
        available_columns = [col for col in columns if col in self.rfm_results.columns]
        return self.rfm_results[available_columns].to_dict('records')
    
    def get_segment_summary(self) -> Dict:
        """
        Get summary statistics for RFM metrics
        
        Returns:
        --------
        Dict
            Summary statistics of RFM metrics
        """
        if self.rfm_results is None:
            self.calculate_rfm_scores()
            
        summary = self.rfm_results.agg({
            'Recency': ['mean', 'min', 'max'],
            'Frequency': ['mean', 'min', 'max'],
            'Monetary': ['mean', 'min', 'max']
        }).round(2)
        
        return {
            'rfm_summary': summary,
            'customer_count': len(self.rfm_results)
        }

if __name__ == "__main__":
    print("Enhanced RFM Analyzer Ready!")