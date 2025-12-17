# Student Performance Prediction System

A machine learning-based web application that predicts student performance (Exam 5) based on previous assessment scores (Exams 1-4) using **Multiple Linear Regression**. This system helps educators identify at-risk students early.

## Features
- **Predictive Model**: Uses Linear Regression (`scikit-learn`) to forecast future scores.
- **Interactive Dashboard**: Modern, responsive web interface built with **Flask**.
- **Risk Analysis**: Automatically categorizes students into risk levels (High, Moderate, Low).
- **Dynamic Updates**: View predictions instantly without page reloads.

## Project Structure
- `app.py`: The Flask application controller.
- `model.py`: Handles model training and prediction logic.
- `data/student_data.csv`: Dataset containing student records.
- `templates/index.html`: The frontend dashboard.
- `static/style.css`: Custom styling for the interface.

## Setup & Running

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   python app.py
   ```

3. **Access the Dashboard**:
   Open your browser and navigate to: `http://127.0.0.1:5000`

## How it Works
1. When the application starts, it trains the model on students who have already completed Exam 5.
2. On the dashboard, click **"Generate Predictions"**.
3. The system predicts scores for students who haven't taken Exam 5 yet and displays their calculated Risk Level.
