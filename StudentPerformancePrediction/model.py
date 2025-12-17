
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sqlalchemy import create_engine
import pymysql

class PerformanceModel:
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.model = None
        self.features = ['Exam1', 'Exam2', 'Exam3', 'Exam4']
        self.target = 'Exam5'
        self.metrics = {}
        
        # Database connection details
        self.db_user = 'root'
        self.db_password = ''  # Update as needed
        self.db_host = 'localhost'
        self.db_name = 'student_db'
        self.connection_string = f"mysql+pymysql://{self.db_user}:{self.db_password}@{self.db_host}/{self.db_name}"
        self.engine = create_engine(self.connection_string)

    def load_data(self):
        """Loads data from MySQL database."""
        try:
            return pd.read_sql('SELECT * FROM students', self.engine)
        except Exception as e:
            print(f"Error loading data from SQL: {e}")
            # Fallback for safety if DB isn't set up yet, though requirement says use DB
            if self.data_path:
                print("Falling back to CSV")
                return pd.read_csv(self.data_path)
            raise e

    def train(self):
        """Trains the linear regression model. Uses fallback weights if no training data exists."""
        df = self.load_data()
        
        train_data = []
        if self.target in df.columns:
            # Split into training (known Exam5) and prediction (unknown Exam5) sets
            train_data = df.dropna(subset=[self.target])
        else:
            print(f"Target column '{self.target}' not found in data. Proceeding with fallback weights.")
        
        self.model = LinearRegression()
        
        if len(train_data) < 2:
            print("Warning: Not enough training data found for Exam5. Using Pre-defined Model Weights.")
            # Fallback: Approximate average of exams (or previous successful weights)
            # Previous run weights: ~0.24, 0.27, 0.23, 0.26
            self.model.coef_ = np.array([0.25, 0.25, 0.25, 0.25])
            self.model.intercept_ = 0.0
            
            self.metrics['r2'] = 0.0  # structured as 0 to indicate no training fit
            self.metrics['coefficients'] = self.model.coef_.tolist()
            self.metrics['intercept'] = self.model.intercept_
            
        else:
            X = train_data[self.features]
            y = train_data[self.target]
            
            self.model.fit(X, y)
            
            # Calculate training metrics
            y_pred = self.model.predict(X)
            self.metrics['r2'] = r2_score(y, y_pred)
            self.metrics['coefficients'] = self.model.coef_.tolist()
            self.metrics['intercept'] = self.model.intercept_
        
        return self.metrics

    def predict(self):
        """Predicts Exam5 for missing rows and calculates risk for ALL students."""
        if not self.model:
            self.train()
            
        df = self.load_data()
        
        # separate known and unknown
        mask_missing = df[self.target].isnull()
        
        # Prepare a column for the final score (Actual or Predicted)
        # Initialize with actual scores
        df['Predicted_Exam5'] = df[self.target]
        
        # Predict for missing
        if mask_missing.any():
            X_pred = df.loc[mask_missing, self.features]
            predictions = self.model.predict(X_pred)
            df.loc[mask_missing, 'Predicted_Exam5'] = np.round(predictions, 1)
            
        # Calculate risk for ALL students
        df['Risk_Level'] = df['Predicted_Exam5'].apply(self._calculate_risk)
        
        # Add a flag to distinguish (optional, useful for UI)
        df['Is_Predicted'] = mask_missing
        
        # Handle NaN for display safety (though logic above should cover it)
        df = df.fillna('')
        
        # Drop the original target if it still has NaNs (it might if we didn't fill it in the DF itself)
        # But here we use 'Predicted_Exam5' for the result. 
        # We can drop the original 'Exam5' col to avoid confusion or keep it.
        # Let's keep 'Student_ID', Exams 1-4, 'Predicted_Exam5', 'Risk_Level'
        
        results = df.to_dict(orient='records')
        return results

    def _calculate_risk(self, score):
        if score < 50:
            return 'High Risk'
        elif score < 75:
            return 'Moderate Risk'
        else:
            return 'Low Risk'

    def get_all_students(self):
        """Returns all student data for display."""
        df = self.load_data()
        
        # Fill NaN for display
        df = df.fillna('')
        
        return df.to_dict(orient='records')

if __name__ == "__main__":
    from sql import import_data_to_db
    
    # Sync DB with CSV first
    print("--- Syncing Database with latest CSV Data ---")
    import_data_to_db()
    
    # This block allows running the file directly to test the model
    model = PerformanceModel()
    try:
        print("--- initializing model training ---")
        metrics = model.train()
        
        print("\n[Model Performance]")
        print(f"R2 Score: {metrics['r2']:.4f}")
        print(f"Coefficients: {metrics['coefficients']}")
        print(f"Intercept: {metrics['intercept']:.4f}")
        
        print("\n[Prediction Preview]")
        results = model.predict()
        # Convert to DataFrame for a nicer terminal display
        if results:
            df_results = pd.DataFrame(results)
            # Show specific columns of interest
            cols_to_show = ['Student_ID', 'Exam1', 'Exam2', 'Exam3', 'Exam4', 'Predicted_Exam5', 'Risk_Level']
            # Handle case if some original columns are missing
            cols_to_show = [c for c in cols_to_show if c in df_results.columns]
            
            print(df_results[cols_to_show].head(10).to_string(index=False))
            print(f"\nTotal Students Processed: {len(df_results)}")
        else:
            print("No data found or predictions made.")
            
    except Exception as e:
        print(f"An error occurred during execution: {e}")
        print("\nTip: Make sure the database is set up. Try running 'python sql.py' first.")
