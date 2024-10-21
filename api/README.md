# Text Classifier API using BioBERT

This API provides access to a text classification model that classifies variables using BioBERT.

## Build and start the docker image

To build the docker image, a token for downloading the model from the Hugging Face Hub must be provided.

```
cd api
docker build --build-arg HF_TOKEN=$HF_TOKEN --tag biobert-api .
docker run -p 5000:5000 biobert-api
```

# Predict
```
http://localhost:5000/predict?variable={your_variable}
```
### Example prediction
```
http://localhost:5000/predict?variable=where do you live?
```

