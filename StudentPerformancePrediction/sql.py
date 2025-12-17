import pandas as pd
from sqlalchemy import create_engine, text
import os

# Database Configuration
# You may need to update these values based on your MySQL configuration
DB_USER = 'root'
DB_PASSWORD = ''  # Add your password here if set
DB_HOST = 'localhost'
DB_NAME = 'student_db'

def create_database():
    """Creates the database if it doesn't exist."""
    # Connect to MySQL server (no specific database yet)
    connection_string = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}"
    engine = create_engine(connection_string)

    try:
        with engine.connect() as conn:
            conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}"))
            print(f"Database '{DB_NAME}' checked/created successfully.")
    except Exception as e:
        print(f"Error creating database: {e}")
        return False
    return True

def import_data_to_db():
    """Reads CSV data and imports it into the MySQL database."""
    csv_path = os.path.join(os.path.dirname(__file__), 'data', 'student_data.csv')
    
    if not os.path.exists(csv_path):
        print(f"Error: Data file not found at {csv_path}")
        return

    # Create engine connecting to the specific database
    connection_string = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
    engine = create_engine(connection_string)

    try:
        # Read CSV
        df = pd.read_csv(csv_path)
        
        # Write to SQL
        # Using 'students' as the table name
        df.to_sql('students', engine, if_exists='replace', index=False)
        print("Data successfully imported from CSV to MySQL table 'students'.")
        
        # Verify
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM students"))
            count = result.scalar()
            print(f"Total rows in 'students' table: {count}")
            
    except Exception as e:
        print(f"Error importing data: {e}")

if __name__ == "__main__":
    print("Setting up database...")
    if create_database():
        import_data_to_db()
