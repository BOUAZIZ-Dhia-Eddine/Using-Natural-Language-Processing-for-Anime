import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from theme_classifier import ThemeClassifier 
import pandas as pd 
from charactere_network import Charactere_network_generator 
from charactere_network import NameEntityRecognizer

def get_themes(theme_list, subtitle_path, save_path):
    list_themes = theme_list.split(',')
    theme_classifier = ThemeClassifier(list_themes)
    
    outputs_df = theme_classifier.get_theme(subtitle_path, save_path)
    list_themes = [theme for theme in list_themes if theme != 'dialogue']
    outputs_df = outputs_df[list_themes]
    outputs_df = outputs_df[list_themes].sum().reset_index()
    outputs_df.columns = ['theme', 'score']
    
    plt.figure(figsize=(5, 2.6))
    plt.barh(outputs_df['theme'], outputs_df['score'], color='blue')
    plt.xlabel('Score')
    plt.title('Themes Score')
    plt.tight_layout()
    
  
    graph_path = 'theme_score_plot.png'
    plt.savefig(graph_path)
    plt.close()  

    return graph_path 


#-------------------------------------------------------------------
def get_chars(subtitles_path,save_path):
    ner= NameEntityRecognizer()
    df=ner.get_ners(save_path)
    chs= Charactere_network_generator()
    df=chs(df)
    return chs.draw_graph(df)
    

    

def main():
    
    with gr.Blocks() as iface:
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Thème Classification (-Zero Shot Classification-)</h1>")
                
                with gr.Row():
                    with gr.Column():
                        plot = gr.Image(label="Themes Score")
                    with gr.Column():
                        theme_list = gr.Textbox(label='Theme')
                        Subtitle_path = gr.Textbox(label='Subtitle_path')
                        save_path = gr.Textbox(label='Save_path')
                        get_themes_butt = gr.Button('Submit')
                        get_themes_butt.click(get_themes, inputs=[theme_list, Subtitle_path, save_path], outputs=[plot])
        with gr.Row():
            with gr.Column():
                gr.HTML("Name Entity Recognition (NER)")
                with gr.Row():
                    with gr.Column():
                         graph=gr.HTML()
                    with gr.Column():
                        subtitles_path=gr.Text('Subtitle Path')
                        save_path=gr.Text('Save Path')
                        submit_button=gr.Button('Submit')
                        submit_button.click(get_chars,inputs=[subtitles_path,save_path],outputs=[graph])
                        
                   
                                

    iface.launch(share=True)

if __name__ == "__main__":
    main()
