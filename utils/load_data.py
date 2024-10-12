from glob import glob 
import pandas as pd 
import os

def load_substiles_dataset(path):
    episods = []
    dialogues = []
    files = glob(os.path.join(path, '*.ass'))
    
    if not files: 
        print(f"folder not exist : {path}")
    
    for file in files:
        print(f"Traitement du fichier : {file}")  
        episods.append(int(file.split('-')[1].split('.')[0]))
        
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            lines = lines[27:]
            rows = [",".join(line.split(',')[9:]) for line in lines]
        rows = [line.replace("\\N", ' ') for line in rows]
        script = " ".join(rows)       
        
        dialogues.append(script)
        df = pd.DataFrame.from_dict({'episode':episods,'script':dialogues})

    return df