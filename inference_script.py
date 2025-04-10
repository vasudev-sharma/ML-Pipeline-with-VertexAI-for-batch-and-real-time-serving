from flask import Flask, request, jsonify
import os
import pickle
import logging

app = Flask(__name__)

# Basic health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Vertex AI requires this endpoint for liveness checks"""
    return jsonify({"status": "healthy"}), 200

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    """Main inference endpoint for Vertex AI online predictions"""
    try:
        instances = request.json.get('instances', [])
        # Load model from Vertex AI's environment variable
        model_path = os.path.join(os.getenv('AIP_MODEL_DIR', '/model'), 'model.pkl')
        # model_path = os.path.join(os.getenv('AIP_STORAGE_DIR', '/model'), 'model.pkl') # TO TEST
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        predictions = [
            model.predict([instance['features']]).tolist()[0]  # Example format
            for instance in instances
        ]
        
        return jsonify({"predictions": predictions})
    
    except Exception as e:
        logging.exception("Prediction failed")
        return jsonify(dict(os.environ)), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
