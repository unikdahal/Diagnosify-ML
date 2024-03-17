import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
from typing import List

app = FastAPI()
pickle_in = open("model.pkl", "rb")
model = pickle.load(pickle_in)

import pickle

with open("disease.pkl", "rb") as file:
    disease = pickle.load(file)


class SymptomData(BaseModel):
    symptom_list: List[str]


def create_input_to_model(symptoms_list, input_symptoms):
    input_to_model = np.zeros(len(symptoms_list), dtype=int)
    for symptom in input_symptoms:
        if symptom in symptoms_list:
            index = symptoms_list.index(symptom)
            input_to_model[index] = 1
    return input_to_model.reshape(1, -1)


def predict_disease_from_symptoms(symptoms, given_symptom_list):
    input_to_model = create_input_to_model(symptoms, given_symptom_list)
    disease_index = model.predict(input_to_model)[0]
    return disease[disease_index]


@app.get("/")
def index():
    return {"message": "Hello , World"}


@app.post("/predict_disease")
def predict(symptoms: SymptomData):
    try:
        df = pd.read_csv("Symptoms_with_order.csv")
        symptoms_list = df["Symptoms"].tolist()
        disease_names = predict_disease_from_symptoms(
            symptoms_list, symptoms.symptom_list
        )
        return {"disease_names": disease_names}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

# uvicorn app:app --reload
