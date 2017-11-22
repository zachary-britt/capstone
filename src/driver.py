import formatter as ft
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import baseline_model

if __name__ == '__main__':

    df = ft.loader_formatter()

    X = df.content.values
    y = df.bias.values

    baseline_model.tfidf_NB_baseline(X, y)
    baseline_model.cos_sim(X, y)
