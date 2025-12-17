import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys

# Add current directory to path to allow importing model
sys.path.append(os.path.dirname(__file__))
from model import PerformanceModel

# Set global style for professional aesthetics
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

def load_data_with_predictions():
    """Loads data and fills missing Exam5 scores using the model's predictions."""
    try:
        # Initialize model
        model = PerformanceModel()
        
        # Get all students (this invokes load_data inside model)
        # We use the predict() method to get the dataframe with predictions
        # predict() returns a list of dicts, we convert to DF
        print("Running model to generate data for visualization...")
        results = model.predict()
        
        if not results:
            print("No data found.")
            return None
            
        df = pd.DataFrame(results)
        
        df = pd.DataFrame(results)
        
        # WE use 'Predicted_Exam5' effectively as 'Exam5' for visualization
        if 'Predicted_Exam5' in df.columns:
            df['Exam 5 (Predicted)'] = df['Predicted_Exam5']
            
        # Convert to numeric just in case
        cols = ['Exam1', 'Exam2', 'Exam3', 'Exam4', 'Exam 5 (Predicted)']
        for col in cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Keep only relevant columns for visualization
        final_cols = [c for c in cols if c in df.columns]
        df = df[final_cols]
        
        print(f"Data prepared (using Predicted Exam 5). Shape: {df.shape}")
        return df
        
    except Exception as e:
        print(f"Error loading/predicting data: {e}")
        return None

def plot_correlation_heatmap(df):
    """Plots a heatmap of correlations between exams."""
    # Drop Student_ID as it's not a feature
    cols = [c for c in df.columns if 'Exam' in c]
    if not cols:
        return
        
    corr_matrix = df[cols].corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f", linewidths=1)
    plt.title('Correlation Matrix of Exams', pad=20)
    plt.tight_layout()
    plt.show()

def plot_histograms(df):
    """Plots the distribution of scores for each exam as histograms."""
    cols = [c for c in df.columns if 'Exam' in c]
    
    plt.figure(figsize=(12, 8))
    for i, col in enumerate(cols):
        if i >= 6: break # Limit just in case
        plt.subplot(2, 3, i+1)
        sns.histplot(df[col], kde=True, bins=10)
        plt.title(f'Distribution of {col}')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

def plot_scatter_with_regression(df):
    """Plots scatter plots with linear regression lines for Exam 1-4 vs Exam 5 (Predicted)."""
    target = 'Exam 5 (Predicted)'
    if target not in df.columns:
        print(f"Skipping regression plot: '{target}' column not found.")
        return

    features = ['Exam1', 'Exam2', 'Exam3', 'Exam4']
    
    plt.figure(figsize=(14, 10))
    for i, feature in enumerate(features):
        if feature not in df.columns: continue
        plt.subplot(2, 2, i+1)
        sns.regplot(data=df, x=feature, y=target, ci=95, line_kws={'color': 'red'}, scatter_kws={'alpha': 0.6})
        plt.title(f'{feature} vs {target}')
        plt.xlabel(f'{feature} Score')
        plt.ylabel(f'{target} Score')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("--- Starting EDA Visualization ---")
    df = load_data_with_predictions()
    
    if df is not None:
        try:
            print("Generating Histograms (Distributions)...")
            plot_histograms(df)
            
            print("Generating Correlation Matrix...")
            plot_correlation_heatmap(df)
            
            print("Generating Scatter Plots with Regression Lines...")
            plot_scatter_with_regression(df)
            
            print("--- EDA Complete. Close plots to finish. ---")
            
        except Exception as e:
            print(f"An error occurred: {e}")
