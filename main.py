from src.data_processing import DataProcessor
from src.rfm_analysis import RFMAnalyzer
import os
import logging
from typing import Optional, Tuple
import pandas as pd

def setup_logging(log_file: Optional[str] = None) -> logging.Logger:
    """
    Configure logging for the application
    
    Parameters:
    -----------
    log_file : str, optional
    Returns:
    --------
    logging.Logger
        Configured logger instance
    """
    # Create formatters and handlers
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    
    # Setup logger
    logger = logging.getLogger('ecommerce_analysis')
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    
    # Add file handler if log file specified
    if log_file:
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

def process_data(data_path: str, logger: logging.Logger) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Process the input data file
    
    Parameters:
    -----------
    data_path : str
        Path to input data file
    logger : logging.Logger
        Logger instance
    
    Returns:
    --------
    Tuple[Optional[pd.DataFrame], Optional[str]]
        Processed DataFrame and error message if any
    """
    try:
        processor = DataProcessor(data_path)
        df = processor.load_data()
        if df is None:
            return None, "Failed to load data"
            
        df_clean = processor.clean_data()
        if df_clean is None:
            return None, "Failed to clean data"
            
        return df_clean, None
        
    except Exception as e:
        return None, str(e)

def run_analysis(df_clean: pd.DataFrame, output_dir: str, logger: logging.Logger) -> bool:
    """
    Run RFM analysis on cleaned data
    
    Parameters:
    -----------
    df_clean : pd.DataFrame
        Cleaned data
    output_dir : str
        Directory to save output files
    logger : logging.Logger
        Logger instance
    
    Returns:
    --------
    bool
        True if analysis successful, False otherwise
    """
    try:
        # Initialize analyzer
        analyzer = RFMAnalyzer(df_clean)
        
        # Calculate RFM scores
        rfm_scores = analyzer.calculate_rfm_scores()
        
        # Save results
        analyzer.save_results(output_dir)
        
        # Log RFM summary
        rfm_summary = analyzer.get_segment_summary()
        logger.info("\nRFM Analysis Summary:")
        logger.info(f"Total Customers: {rfm_summary['customer_count']:,}")
        logger.info("\nMetrics Summary:")
        logger.info("\n" + str(rfm_summary['rfm_summary']))
        
        # Log sales summary
        dashboard_data = analyzer.prepare_dashboard_data()
        if dashboard_data and 'sales_summary' in dashboard_data:
            logger.info("\nSales Summary:")
            for key, value in dashboard_data['sales_summary'].items():
                logger.info(f"{key}: {value:,.2f}")
            
        return True
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        return False

def main():
    # Setup directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    log_dir = os.path.join(base_dir, 'logs')
    
    # Create necessary directories
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(log_dir, 'analysis.log')
    logger = setup_logging(log_file)
    
    try:
        # Input/output paths
        data_path = os.path.join(data_dir, 'data.csv')
        
        # Validate input file
        if not os.path.exists(data_path):
            logger.error(f"Data file not found: {data_path}")
            return
            
        # Process data
        logger.info("Starting data processing...")
        df_clean, error = process_data(data_path, logger)
        if error:
            logger.error(f"Data processing failed: {error}")
            return
            
        logger.info(f"Data processed successfully - Shape: {df_clean.shape}")
        
        # Run analysis
        logger.info("Starting RFM analysis...")
        success = run_analysis(df_clean, data_dir, logger)
        
        if success:
            logger.info("Analysis completed successfully!")
        else:
            logger.error("Analysis failed!")
    
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        logger.error("Stack trace:", exc_info=True)

if __name__ == "__main__":
    main()
