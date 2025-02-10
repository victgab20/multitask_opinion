import pickle
import torch
import os
import pandas as pd

import re
import string

import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from polaridade_model import train_model,train_and_validate, evaluate_model, PolarityClassifier
from dotenv import load_dotenv
import os

MODEL_PATH = "modelo_treinado.pth"

load_dotenv()
path = os.getenv("PATH")

def load_data(nome_arquivo):
    with open(f"{path}\\{nome_arquivo}", "rb") as f:
        return pickle.load(f)

train_apps = load_data("train_apps.pkl")
dev_apps = load_data("dev_apps.pkl")
test_apps = load_data("test_apps.pkl")

train_filmes = load_data("train_filmes.pkl")
dev_filmes = load_data("dev_filmes.pkl")
test_filmes = load_data("test_filmes.pkl")


def preprocess_text(text: str):
    text = text.lower()
    text = re.sub(r'[0-9]+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def pre_processing(df: pd.DataFrame):

    df = df.loc[lambda x: x["stars"] != 3]
    df["stars"] = df["stars"].apply(lambda x: 1 if x > 3 else 0)

    df["text"] = df["text"].apply(preprocess_text)
    df = df[df['text'].notnull() & (df['text'] != '')]
    return df

def tokenize_texts(df: pd.DataFrame):
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
    X = vectorizer.fit_transform(df["text"]).toarray()
    y = df["stars"].values
    return X, y, vectorizer


a = pre_processing(train_apps[["stars", "text"]])
b = pre_processing(dev_apps[["stars", "text"]])
c = pre_processing(test_apps[["stars","text"]])
X_train, y_train, vectorizer = tokenize_texts(a)
X_dev, y_dev, vectorizer = tokenize_texts(b)
X_test, y_test, vectorizer = tokenize_texts(c)

def main():

    if os.path.exists(MODEL_PATH):
        modelo_treinado = PolarityClassifier(input_size=X_train.shape[1])
        modelo_treinado.load_state_dict(torch.load(MODEL_PATH))
        modelo_treinado.eval()
        print("Modelo carregado com sucesso!")
    else:
        print("Nenhum modelo salvo encontrado. Treinando um novo...")

        modelo_treinado = train_and_validate(X_train, y_train, X_dev, y_dev)

        torch.save(modelo_treinado.state_dict(), MODEL_PATH)
        print(f"Modelo treinado e salvo em {MODEL_PATH}!")

    evaluate_model(modelo_treinado, X_test, y_test)

if __name__ == "__main__":
    main()
