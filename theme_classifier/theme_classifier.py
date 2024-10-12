from transformers import pipeline 
import os 
import nltk
import torch
import pandas as pd 

from nltk.tokenize import sent_tokenize
import numpy as np 
import sys 
import pathlib 
nltk.download('punkt') 
nltk.download('punkt_tab')

folder_path = pathlib.Path(__file__).parent.resolve()  
sys.path.append(os.path.join(folder_path, "../")) 

from utils import load_substiles_dataset 

class ThemeClassifier: 
    def __init__(self, list_theme, model_task="zero-shot-classification", device=-1):
        self.list_theme = list_theme
        self.model_name = "facebook/bart-large-mnli"
        self.device = device
        self.model_task = model_task
        self.model_classifier = self.load_classifier(self.device)
    
    def load_classifier(self, device):
        classifier = pipeline(
            self.model_task,
            model=self.model_name,
            device=device,
        )
        return classifier 
    
    def get_theme_inference(self, df_p, batch_number=20):
        script_batch = []
        theme_score_batch = {}
        
      
        episode = sent_tokenize(df_p)
        
      
        for i in range(0, len(episode), batch_number):
            batch_sentences = " ".join(episode[i:i + batch_number]) 
            script_batch.append(batch_sentences)
            
      
        outputs = self.model_classifier(script_batch, self.list_theme, multi_label=True)
        
       
        for output in outputs:
            for label, score in zip(output['labels'], output['scores']):
                if label not in theme_score_batch:
                    theme_score_batch[label] = []
                theme_score_batch[label].append(score)
                
        theme_score_batch = {key: np.mean(np.array(value)) for key, value in theme_score_batch.items()}
            
        return theme_score_batch 
    

    
    def get_theme(self,path_dataset,save_path=None):
        
        if save_path is not None and os.path.exists(save_path):
            df_p=pd.read_csv(save_path)
            return df_p

        df_p = load_substiles_dataset(path_dataset)

        outputs_theme=df_p['script'].apply(self.get_theme_inference)
        outputs_theme=pd.DataFrame(outputs_theme.to_list())
        df_p = pd.concat([df_p, outputs_theme], axis=1)
        if save_path is not None :
            df_p.to_csv(save_path,index=False)
        return df_p
