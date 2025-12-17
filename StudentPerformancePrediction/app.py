
from flask import Flask, render_template, jsonify
from model import PerformanceModel
import os

app = Flask(__name__)

# Initialize model
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'student_data.csv')
model = PerformanceModel(DATA_PATH)

@app.route('/')
def index():
    # Train model to get stats
    metrics = model.train()
    
    # Get all students to display raw data if needed, or just partial
    all_students = model.get_all_students()
    
    return render_template('index.html', metrics=metrics, students=all_students)

@app.route('/predict')
def predict():
    predictions = model.predict()
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)
