import os
from flask import Flask, request, jsonify
import torch
import csv
from transformers import AutoModel, pipeline

app = Flask(__name__)

cuda_available = torch.cuda.is_available()

# import tags mapped to label and iri
mapped_tags = {}
with open('mappedAnnotations.csv', mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        mapped_tags[row['tag']] = {
            "label": row['label'],
            "iri": row['iri'],
            "domain": row['domain']
        }

HF_TOKEN = os.environ.get("HF_TOKEN")
pipe = pipeline("text-classification", model="JuSas/biobert-Maelstrom-cleaned", top_k=None, use_auth_token=HF_TOKEN)

@app.route('/')
def home():
    return jsonify({"message": "JSON API to predict Maelstrom tags using a fine-tuned BioBERT model"})

@app.route('/predict')
def predict():
    variable = request.args.get('variable')

    if variable is None:
        print('Variable is None')
        return jsonify({'errorCode': 404, 'message': 'Variable not found'})
    else:
        result = pipe(variable)
        result_sorted = sorted(result[0], key=lambda x: x['score'], reverse=True)

        mapped_result = []
        print("res",result_sorted)
        print("mapp", mapped_tags)
        for res in result_sorted:

            mapped_result.append(
                {
                    "tag": res["label"],
                    "confidence": round(res["score"] * 100, 5),
                    "label": mapped_tags[res["label"]]['label'],
                    "iri": mapped_tags[res["label"]]['iri'],
                    "domain": mapped_tags[res['label']]['domain']
                }
            )
        return jsonify({'variable': variable, 'prediction': mapped_result})


if __name__ == '__main__':
    app.run(debug=False, port=5000, host="0.0.0.0")
