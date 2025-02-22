# E-Commerce Customer Analysis Project

A comprehensive analytics solution for e-commerce customer behavior analysis, segmentation, and RFM (Recency, Frequency, Monetary) analysis.

## Features

- **Data Processing**: Robust cleaning and preprocessing of e-commerce transaction data
- **RFM Analysis**: Customer scoring based on Recency, Frequency, and Monetary values
- **Customer Segmentation**: K-means clustering for customer behavior analysis
- **Visualization**: Multiple visualization options for analysis results
- **Quality Reports**: Detailed data quality and analysis reports

Project Structure
``` 
ecommerce_analysis/
├── data/                      # Data files and analysis outputs
│   ├── customer_clusters.csv
│   ├── customer_segments.csv
│   ├── dashboard_data.json
│   ├── data_quality_report.json
│   └── data.csv
├── logs/                      # Log files
├── src/                       # Source code
│   ├── __init__.py
│   ├── customer_segmentation.py
│   ├── data_processing.py
│   └── rfm_analysis.py
├── visualizations/            # Visualization HTML files
│   ├── geographic_distribution.html
│   ├── rfm_distributions.html
│   └── temporal_patterns.html
├── main.py                    # Main execution script
└── requirements.txt           # Project dependencies
``` 
    
Requirements

Python 3.8+
Dependencies listed in requirements.txt:

pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=0.24.0
scipy>=1.7.0



Installation

Create and activate a virtual environment:

bashCopypython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies:

bashCopypip install -r requirements.txt
Usage

Place your input data file (CSV format) in the data directory as data.csv
Run the analysis:

bashCopypython main.py

Check the data directory for outputs:


customer_segments.csv: RFM analysis results
customer_clusters.csv: Customer segmentation results
dashboard_data.json: Aggregated metrics for visualization
Various visualization files (PNG, HTML)

Input Data Format
The input CSV file should contain the following columns:

CustomerID
InvoiceNo
InvoiceDate
StockCode
Description
Quantity
UnitPrice
Country

Analysis Components
Data Processing (data_processing.py)

Data cleaning and validation
Feature engineering
Time-based metrics calculation
Customer feature aggregation

RFM Analysis (rfm_analysis.py)

Recency, Frequency, and Monetary value calculation
Customer scoring
Segment analysis
Dashboard data preparation

Customer Segmentation (customer_segmentation.py)

K-means clustering
Customer behavior analysis
Segment interpretation
Visualization generation

Output Files
CSV Files

customer_segments.csv: Customer-level RFM metrics and scores
customer_clusters.csv: Cluster assignments and characteristics

Visualizations

cluster_characteristics.png: Cluster feature distributions
elbow_curve.png: K-means optimal cluster analysis
segment_distribution.png: Customer segment distribution
segment_features.png: Segment feature importance

Reports

dashboard_data.json: Aggregated metrics for dashboards
data_quality_report.json: Data quality metrics and validation results

Logging

Application logs are stored in logs/analysis.log
Includes data processing steps, validation results, and analysis outcomes
Configurable log levels and formats

Data Quality Checks
The project implements various data quality checks:

Missing value detection and handling
Outlier detection and removal
Data type validation
Feature value range validation
Duplicate detection

Author
Created by: Tao ZHANG

Last Updated: February 2025
