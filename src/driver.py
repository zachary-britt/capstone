import formatter
import pandas as pd
import numpy as np
import baseline_model

if __name__ == '__main__':

    df = formatter.main()

    X = df.content.values
    d = df.date.values
    y = df.bias.values

    

    baseline_model.tfidf_NB_baseline(X, y)
    baseline_model.cos_sim(X, y)
