
import unittest
import os
import json
from app import app
from model import PerformanceModel

class TestStudentPerformance(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        self.data_path = os.path.join(os.path.dirname(__file__), 'data', 'student_data.csv')
        self.model = PerformanceModel(self.data_path)

    def test_model_training(self):
        """Test if model trains and produces metrics."""
        metrics = self.model.train()
        self.assertIn('r2', metrics)
        self.assertTrue(metrics['r2'] > 0) # Should have some correlation
        print(f"\nModel R2 Score: {metrics['r2']}")

    def test_prediction_logic(self):
        """Test if prediction returns results for missing Exam5."""
        self.model.train()
        predictions = self.model.predict()
        self.assertTrue(len(predictions) > 0)
        first_pred = predictions[0]
        self.assertIn('Predicted_Exam5', first_pred)
        self.assertIn('Risk_Level', first_pred)
        print(f"\nSample Prediction: {first_pred}")

    def test_index_route(self):
        """Test if home page loads."""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Student Performance Dashboard', response.data)

    def test_predict_route(self):
        """Test prediction API endpoint."""
        response = self.app.get('/predict')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(len(data) > 0)

if __name__ == '__main__':
    unittest.main()
