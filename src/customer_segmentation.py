import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from typing import Dict, Optional
import logging
import matplotlib.pyplot as plt
import seaborn as sns

class CustomerSegmentation:
    """Customer segmentation using RFM metrics and K-means clustering"""
    
    def __init__(self, rfm_data: pd.DataFrame, logger: Optional[logging.Logger] = None):
        """
        Initialize Customer Segmentation
        
        Parameters:
        -----------
        rfm_data : pd.DataFrame
            RFM analysis results with customer metrics
        logger : logging.Logger, optional
            Logger instance for tracking progress
        """
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info(f"Input data shape: {rfm_data.shape}")
        
        # Store original data
        self.original_data = rfm_data.copy()
        
        # Define features for clustering
        self.features = ['Recency', 'Frequency', 'Monetary']
        
        # Validate and prepare data
        self._validate_data()
        self.data = self._prepare_data()
        self.segments = None
        
    def _validate_data(self) -> None:
        """Validate input data has required columns"""
        missing = [col for col in self.features if col not in self.original_data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
            
    def _prepare_data(self) -> pd.DataFrame:
        """Prepare data for clustering"""
        df = self.original_data[['CustomerID'] + self.features].copy()
        
        # Handle any missing or infinite values
        for feature in self.features:
            df[feature] = df[feature].replace([np.inf, -np.inf], np.nan)
            if df[feature].isnull().any():
                median = df[feature].median()
                df[feature].fillna(median, inplace=True)
                self.logger.info(f"Filled {feature} null values with median: {median}")
        
        return df
        
    def _find_optimal_clusters(self, X: np.ndarray, max_k: int = 10) -> int:
        """Find optimal number of clusters using elbow method"""
        inertias = []
        K = range(1, max_k + 1)
        
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
        
        # Plot elbow curve
        plt.figure(figsize=(10, 6))
        plt.plot(K, inertias, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Inertia')
        plt.title('Elbow Method For Optimal k')
        plt.savefig('data/elbow_curve.png')
        plt.close()
        
        # Find elbow point using the point of maximum curvature
        diffs = np.diff(inertias, 2)
        elbow_point = np.argmin(diffs) + 2
        
        return min(elbow_point, max_k)
        
    def perform_segmentation(self, n_clusters: Optional[int] = None) -> pd.DataFrame:
        """
        Perform customer segmentation using K-means clustering
        
        Parameters:
        -----------
        n_clusters : int, optional
            Number of clusters. If None, will be determined automatically
            
        Returns:
        --------
        pd.DataFrame
            Original data with cluster assignments and cluster interpretations
        """
        try:
            # Scale features
            scaler = RobustScaler()
            X = scaler.fit_transform(self.data[self.features])
            
            # Determine number of clusters if not specified
            if n_clusters is None:
                n_clusters = self._find_optimal_clusters(X)
                self.logger.info(f"Determined optimal number of clusters: {n_clusters}")
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            
            # Analyze clusters and get interpretations
            centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
            cluster_summary = pd.DataFrame(centers_original, columns=self.features)
            
            # Create cluster interpretations
            cluster_interpretations = []
            for cluster in range(n_clusters):
                metrics = cluster_summary.loc[cluster]
                interpretation = ""
                
                # Determine cluster type based on metrics
                if metrics['Recency'] < cluster_summary['Recency'].mean() and metrics['Frequency'] > cluster_summary['Frequency'].mean() and metrics['Monetary'] > cluster_summary['Monetary'].mean():
                    interpretation = "Best Customers"
                elif metrics['Recency'] > cluster_summary['Recency'].mean() and metrics['Frequency'] < cluster_summary['Frequency'].mean():
                    interpretation = "Lost Customers"
                elif metrics['Recency'] < cluster_summary['Recency'].mean() and metrics['Monetary'] > cluster_summary['Monetary'].mean():
                    interpretation = "Big Spenders"
                elif metrics['Frequency'] > cluster_summary['Frequency'].mean():
                    interpretation = "Loyal Customers"
                elif metrics['Recency'] < cluster_summary['Recency'].mean():
                    interpretation = "New Customers"
                else:
                    interpretation = "Average Customers"
                
                cluster_interpretations.append(interpretation)
            
            # Merge results back to original data
            result = self.original_data.copy()
            result['Cluster'] = cluster_labels
            result['Cluster_Type'] = [cluster_interpretations[c] for c in cluster_labels]
            
            # Add cluster metrics
            for feature in self.features:
                col_name = f'Cluster_Avg_{feature}'
                result[col_name] = result['Cluster'].map(
                    cluster_summary[feature]
                )
            
            # Analyze clusters
            self._analyze_clusters(kmeans.cluster_centers_, scaler)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Segmentation failed: {str(e)}")
            raise
            
    def _analyze_clusters(self, centers: np.ndarray, scaler: RobustScaler) -> None:
        """Analyze and visualize cluster characteristics"""
        # Transform centers back to original scale
        centers_original = scaler.inverse_transform(centers)
        
        # Create cluster summary
        summary = pd.DataFrame(
            centers_original,
            columns=self.features
        )
        summary.index.name = 'Cluster'
        
        # Log cluster characteristics
        self.logger.info("\nCluster Centers:")
        self.logger.info(summary.round(2))
        
        # Save cluster sizes
        sizes = self.data['Cluster'].value_counts().sort_index()
        self.logger.info("\nCluster Sizes:")
        self.logger.info(sizes)
        
        # Interpret clusters
        self._interpret_clusters(summary, sizes)
        
        # Create visualization
        self._plot_cluster_characteristics(summary)
        
    def _interpret_clusters(self, centers: pd.DataFrame, sizes: pd.Series) -> None:
        """Provide business interpretation of clusters"""
        # Calculate relative metrics for each cluster
        relative_metrics = centers.copy()
        for col in centers.columns:
            relative_metrics[col] = (centers[col] - centers[col].mean()) / centers[col].std()
        
        # Interpret each cluster
        self.logger.info("\nCluster Interpretations:")
        total_customers = sizes.sum()
        
        for cluster in centers.index:
            metrics = relative_metrics.loc[cluster]
            size = sizes[cluster]
            percentage = (size / total_customers) * 100
            
            interpretation = []
            # Basic size information
            interpretation.append(f"\nCluster {cluster} ({size:,} customers, {percentage:.1f}% of total):")
            
            # RFM characteristics
            if metrics['Recency'] < -0.5:
                interpretation.append("- Recent customers (lower recency score)")
            elif metrics['Recency'] > 0.5:
                interpretation.append("- Less recent customers (higher recency score)")
                
            if metrics['Frequency'] > 0.5:
                interpretation.append("- High frequency shoppers")
            elif metrics['Frequency'] < -0.5:
                interpretation.append("- Low frequency shoppers")
                
            if metrics['Monetary'] > 0.5:
                interpretation.append("- High spenders")
            elif metrics['Monetary'] < -0.5:
                interpretation.append("- Low spenders")
            
            # Overall segment interpretation
            if metrics['Recency'] < -0.5 and metrics['Frequency'] > 0.5 and metrics['Monetary'] > 0.5:
                interpretation.append("→ Best Customers: Recent, frequent buyers with high spending")
            elif metrics['Recency'] < 0 and metrics['Frequency'] > 0 and metrics['Monetary'] > 0:
                interpretation.append("→ Loyal Customers: Relatively recent and regular buyers")
            elif metrics['Recency'] < -0.5 and metrics['Frequency'] < 0 and metrics['Monetary'] > 0:
                interpretation.append("→ Big Spenders: Recent big ticket buyers but less frequent")
            elif metrics['Recency'] > 0.5 and metrics['Frequency'] > 0 and metrics['Monetary'] > 0:
                interpretation.append("→ Lost Valuable Customers: Previous regular customers who haven't returned")
            elif metrics['Recency'] > 0.5 and metrics['Frequency'] < 0 and metrics['Monetary'] < 0:
                interpretation.append("→ Lost Customers: Previous occasional buyers who haven't returned")
            elif metrics['Recency'] < 0 and metrics['Frequency'] < 0 and metrics['Monetary'] < 0:
                interpretation.append("→ New/Low-Value Customers: Recent but infrequent and low-value purchases")
            
            # Add actual values
            interpretation.append("\nActual Values:")
            for metric in self.features:
                interpretation.append(f"- {metric}: {centers.loc[cluster, metric]:.2f}")
            
            self.logger.info("\n".join(interpretation))
            
        # Add recommendations
        self.logger.info("\nRecommended Actions:")
        self.logger.info("1. Best/Loyal Customers: Reward and engage with loyalty programs")
        self.logger.info("2. Big Spenders: Encourage more frequent visits with personalized offers")
        self.logger.info("3. Lost Valuable Customers: Re-engagement campaign with special incentives")
        self.logger.info("4. Lost Customers: Survey to understand churn reasons")
        self.logger.info("5. New/Low-Value Customers: Nurture with entry-level promotions")
        
    def _plot_cluster_characteristics(self, centers: pd.DataFrame) -> None:
        """Create visualization of cluster characteristics"""
        plt.figure(figsize=(12, 6))
        
        # Heatmap of cluster centers
        sns.heatmap(
            centers,
            annot=True,
            fmt='.2f',
            cmap='YlOrRd',
            cbar_kws={'label': 'Value'}
        )
        
        plt.title('Cluster Characteristics')
        plt.tight_layout()
        plt.savefig('data/cluster_characteristics.png')
        plt.close()

def main():
    """Main function for testing"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Setup directories
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        # Define input/output paths
        input_file = os.path.join(data_dir, 'customer_segments.csv')
        output_file = os.path.join(data_dir, 'customer_clusters.csv')
        
        # Check if input file exists
        if not os.path.exists(input_file):
            logger.error(f"Input file not found: {input_file}")
            logger.info("Please run the RFM analysis first to generate the customer segments file.")
            return
        
        # Load RFM results
        logger.info(f"Loading RFM data from: {input_file}")
        rfm_data = pd.read_csv(input_file)
        
        # Perform segmentation
        segmentation = CustomerSegmentation(rfm_data, logger)
        results = segmentation.perform_segmentation()
        
        # Save results
        try:
            # Save full results with cluster assignments
            results.to_csv(output_file, index=False)
            logger.info(f"Cluster analysis results saved to: {output_file}")
            
            # Save cluster summary
            summary_file = os.path.join(data_dir, 'cluster_summary.csv')
            cluster_summary = results.groupby(['Cluster', 'Cluster_Type']).agg({
                'CustomerID': 'count',
                'Recency': 'mean',
                'Frequency': 'mean',
                'Monetary': 'mean'
            }).round(2)
            cluster_summary.to_csv(summary_file)
            logger.info(f"Cluster summary saved to: {summary_file}")
            
            logger.info("\nCluster Analysis Summary:")
            for cluster_type in results['Cluster_Type'].unique():
                count = (results['Cluster_Type'] == cluster_type).sum()
                percentage = (count / len(results) * 100)
                logger.info(f"{cluster_type}: {count:,} customers ({percentage:.1f}%)")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise
        
    except Exception as e:
        logger.error(f"Segmentation failed: {str(e)}")
        raise

if __name__ == '__main__':
    main()