# main.py - FastAPI application for DGA domain detection
import os
import pickle
import numpy as np
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pydantic import BaseModel
import uvicorn

# Define the custom AttentionLayer class
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                               initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                               initializer="zeros")
        super(AttentionLayer, self).build(input_shape)
        
    def call(self, x):
        # Calculate attention scores
        e = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        
        # Get attention weights
        a = tf.nn.softmax(e, axis=1)
        
        # Apply attention weights to the input
        output = x * a
        
        # Sum over the sequence dimension
        return tf.reduce_sum(output, axis=1)
    
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]
    
    # Add get_config method for serialization
    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        return config

# Create FastAPI app
app = FastAPI(
    title="DGA Domain Detection API",
    description="API for detecting Domain Generation Algorithm (DGA) domains using LSTM+Attention model",
    version="1.0.0"
)

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
if not os.path.exists("templates"):
    os.makedirs("templates")

if not os.path.exists("static"):
    os.makedirs("static")

app.mount("/static", StaticFiles(directory="static"), name="static")

# Define request and response models
class DomainRequest(BaseModel):
    domain: str

class DomainResponse(BaseModel):
    domain: str
    prediction: str
    is_dga: bool
    confidence: float

# Load model and preprocessing components
@app.on_event("startup")
async def load_detection_model():
    global model, tokenizer, max_length
    
    try:
        print("Loading model and preprocessing components...")
        
        # Load the model with custom objects
        model_path = "../ML_v2/dga_lstm_attention_model.h5"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        # Register the custom layer
        custom_objects = {"AttentionLayer": AttentionLayer}
        model = load_model(model_path, custom_objects=custom_objects)
        
        # Load tokenizer
        tokenizer_path = "../ML_v2/dga_tokenizer.pickle"
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}")
            
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        
        # Load config
        config_path = "../ML_v2/dga_config.pickle"
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
            
        with open(config_path, 'rb') as handle:
            config = pickle.load(handle)
            max_length = config['max_length']
        
        print("Model and components loaded successfully!")
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise e

# Predict function
def predict_domain(domain):
    # Preprocess domain
    domain = domain.lower()
    sequence = tokenizer.texts_to_sequences([domain])
    padded = pad_sequences(sequence, maxlen=max_length, padding='post')
    
    # Make prediction
    prediction = float(model.predict(padded, verbose=0)[0][0])
    is_dga = prediction > 0.5
    confidence = prediction if is_dga else 1 - prediction
    
    return {
        "domain": domain,
        "prediction": "DGA" if is_dga else "Legitimate",
        "is_dga": is_dga,
        "confidence": float(confidence)
    }

# API routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page with a form to test domain detection"""
    # Create an HTML template if it doesn't exist
    if not os.path.exists("templates/index.html"):
        with open("templates/index.html", "w") as f:
            f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>DGA Domain Detection</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { padding: 20px; }
        .result-card { margin-top: 20px; }
        .confidence-bar { height: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="mb-4">DGA Domain Detection</h1>
        <div class="card">
            <div class="card-body">
                <h5 class="card-title">Check if a domain is generated by a DGA</h5>
                <form id="domain-form" method="post" action="/check">
                    <div class="mb-3">
                        <label for="domain" class="form-label">Domain Name:</label>
                        <input type="text" class="form-control" id="domain" name="domain" required
                               placeholder="Enter a domain name (baidu.com、google.com、cvyh1po636avyrsxebwbkn7.ddns.net)">
                    </div>
                    <button type="submit" class="btn btn-primary">Check Domain</button>
                </form>
            </div>
        </div>

        {% if result %}
        <div class="card result-card">
            <div class="card-body">
                <h5 class="card-title">Detection Result</h5>
                <p><strong>Domain:</strong> {{ result.domain }}</p>
                <p><strong>Classification:</strong> 
                    {% if result.is_dga %}
                    <span class="badge bg-danger">DGA</span>
                    {% else %}
                    <span class="badge bg-success">Legitimate</span>
                    {% endif %}
                </p>
                <p><strong>Confidence:</strong> {{ "%.2f"|format(result.confidence*100) }}%</p>
                <div class="progress">
                    <div class="progress-bar {% if result.is_dga %}bg-danger{% else %}bg-success{% endif %}" 
                         role="progressbar" 
                         style="width: {{ result.confidence*100 }}%" 
                         aria-valuenow="{{ result.confidence*100 }}" 
                         aria-valuemin="0" 
                         aria-valuemax="100">
                        {{ "%.2f"|format(result.confidence*100) }}%
                    </div>
                </div>
            </div>
        </div>
        {% endif %}
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
            """)
    
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/check", response_class=HTMLResponse)
async def check_form(request: Request, domain: str = Form(...)):
    """Handle form submission and render results"""
    result = predict_domain(domain)
    return templates.TemplateResponse("index.html", {"request": request, "result": result})

@app.post("/api/detect", response_model=DomainResponse)
async def detect_domain(request: DomainRequest):
    """API endpoint for domain detection"""
    if not request.domain or len(request.domain.strip()) == 0:
        raise HTTPException(status_code=400, detail="Domain cannot be empty")
    
    return predict_domain(request.domain)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}

# Batch processing endpoint
@app.post("/api/batch-detect")
async def batch_detect(domains: list[str]):
    """Process multiple domains at once"""
    if not domains or len(domains) == 0:
        raise HTTPException(status_code=400, detail="Domain list cannot be empty")
    
    if len(domains) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 domains allowed per request")
    
    results = []
    for domain in domains:
        if domain and len(domain.strip()) > 0:
            results.append(predict_domain(domain))
    
    return {
        "total": len(results),
        "results": results
    }

def main():
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
