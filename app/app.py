from typing import Optional

import joblib
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from email_provider.imap import get_emails

app = FastAPI(title="Phishing Email Model")

# Enable CORS just in case frontend runs separately during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load the scaler since the scaler should be the SAME one used for training
scaler = joblib.load("./models/scaler.pkl")

# feature order ensures training data matches correct paramsi
feature_order = joblib.load("./models/feature_order.pkl")

# load the model
classifier = keras.models.load_model("./models/classifier.keras")

# endpoint to get the emails
@app.get("/emails")
def emails():
    return get_emails(
        host="imap.gmail.com",
        username="", # supply these yourself
        password="" # supply these yourself
    )


# define a request format that the user will pass
class EmailRequest(BaseModel):
    subject: str
    body: str
    sender_email: str
    sender_display_name: Optional[str] = None
    sent_datetime: str

# endpoint for the prediction model
@app.post("/predict")
def predict(request: EmailRequest):
    from components.bert import preprocessor, embedding_model
    from components.numeric import extract_numeric_features
    text = (
            request.subject +
            " [SEP] " +
            request.body
    )
    tokens = preprocessor(tf.constant([text]))
    embedding = embedding_model(tokens, training=False).numpy()

    numeric_features = extract_numeric_features(request)
    numeric = pd.DataFrame(
        [numeric_features],
        columns=scaler.feature_names_in_
    )

    numeric = scaler.transform(numeric)

    features = np.concatenate(
        [
            embedding,
            numeric
        ],
        axis=1
    )
    prediction = classifier(
        features,
        training=False
    ).numpy()

    predicted_class = int(np.argmax(prediction))
    confidence = float(
        prediction[0][predicted_class]
    )

    return {
        "prediction": predicted_class,
        "confidence": confidence
    }


# Mount the static folder so the frontend HTML/CSS/JS is served directly by FastAPI
app.mount("/", StaticFiles(directory="static", html=True), name="static")
