import spacy
import pandas as pd
import numpy as np
from spacy.util import minibatch, compounding
from pathlib import Path
import plac
import os
DATA_PATH = os.environ['DATA_PATH']

import ipdb


def load_text_dfs(data_loc):
    pass



def main(   data_loc=DATA_PATH+'formatted_arts.pkl'
            model_loc=PROJ_PATH+''):
    pass

if __name__ == '__main__':
    plac.call(main)
