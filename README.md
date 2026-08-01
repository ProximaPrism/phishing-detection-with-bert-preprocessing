# Phishing Detection DL Classifier Model with BERT Tokenizer

This is a phishing detection deep learning (DL) classifier model using BERT for tokenization and embedding of text within emails. 

## Project Information

This project uses the **GPU release** of `tensorflow==2.20.x`,
as running the BERT preprocessors can **take hours** on the CPU.

If needed, [embeddings](/embeddings) have been provided if you don't want to spend time on the compute 
or lack a GPU to run the preprocessors.

This project has a simple [email client](/app) using `FastAPI` to show how the model can be used. 
To test, run `uvicorn app:app --reload` within the `/app` directory.

## Running the Project

It is highly recommended you run a seperate virtual environment (`.venv`) with **Python 3.12.x** within the project.

This is because the BERT preprocessors used from `keras_hub` require the `tensorflow-text` library,
which the latest version (as of Jul '26) only works with `tensorflow==2.20.x`

## Model Architecture

![Model Architecture](shared/model_arch.png)

## References

### Training Data

**AVN Phishing Email Classification Dataset**\
From AVN Bluefox (Amritha V Nair) - 2025. Under CC BY 4.0 License\
https://www.kaggle.com/datasets/avnbluefox/avn-phishing-email-classification-dataset?select=AVN_Corpus.csv