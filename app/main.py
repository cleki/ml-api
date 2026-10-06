from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(
    title="Iris Classification API",
    description="A simple API to predict Iris flower species using machine learning.",
    version="1.0.0"
)

# Load the pre-trained model and define class labels
model = joblib.load("model/model.joblib")
CLASSES = ["setosa", "versicolor", "virginica"]

# Define the expected JSON payload schema
class Features(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def home():
    """
    Root endpoint to verify the server is running.
    Redirects user guidance to the interactive documentation.
    """
    return {
        "message": "Iris Classification API is up and running successfully.",
        "documentation": "Visit /docs for the interactive Swagger UI."
    }

@app.get("/health")
def health():
    """
    Health check endpoint for monitoring tools.
    """
    return {"status": "ok"}

@app.post("/predict")
def predict(f: Features):
    """
    Accepts flower features and returns the predicted class index and species name.
    """
    # Convert Pydantic feature attributes into a 2D array structure for scikit-learn
    feature_input = [[f.sepal_length, f.sepal_width, f.petal_length, f.petal_width]]
    
    # Run inference and extract the single prediction scalar
    prediction_index = int(model.predict(feature_input)[0])
    
    return {
        "prediction": prediction_index, 
        "label": CLASSES[prediction_index]
    }