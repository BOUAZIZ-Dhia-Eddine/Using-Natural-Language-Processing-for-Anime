import spacy
from nltk.tokenize import sent_tokenize
import os
import sys
import pandas as pd
import pkg_resources
import pathlib

folder_path = pathlib.Path().parent.resolve()
sys.path.append(os.path.join(folder_path, "../"))

from utils import load_substiles_dataset
from ast import literal_eval

class NameEntityRecognizer:
    def __init__(self, path_data='../data/Subtitles/'):
        self.path = path_data
        self.model = self.load_model()

    def load_model(self):
        nlp = spacy.load("en_core_web_trf")
        return nlp

    def name_recognition(self, data):
        sentences = sent_tokenize(data)
        list_names = list()
        for token in sentences:
            names = set()
            output = self.model(token)
            for doc in output.ents:
                if doc.label_ == 'PERSON':
                    names.add(doc.text.split()[0].upper())  # Indentation corrected here
            list_names.append(names)
        return list_names

    def get_ners(self, save_path=None):
        if save_path is not None and os.path.exists(save_path):
            df = pd.read_csv(save_path)
            df['ners'] = df['ners'].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)
            return df
        
        df = load_substiles_dataset(self.path)
        df['ners'] = df['script'].apply(self.name_recognition)

        if save_path is not None:
            df.to_csv(save_path, index=False)
            print(f'Data saved in {save_path}')
        return df
