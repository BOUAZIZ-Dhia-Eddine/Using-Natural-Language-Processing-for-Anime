from pyvis.network import network
import networkx as nx
import pandas as pd

class Charactere_network_generator():
    def __init__(self):
        pass

    def calcul_occurrences(self, df, window=10):
        combinaison = {}

        for ligne in df['ners']:
            ligne = [elem for item in ligne if item for elem in item]
            for i in range(len(ligne)):
                j = i - window
                if i == (window - 1):
                    for k in range(0, window - 1):
                        if ligne[k] != ligne[k + 1]:
                            if k + 1 < window - 1:
                                x = tuple(sorted([ligne[k], ligne[k + 1]]))
                                print(x)
                                if x in combinaison:
                                    combinaison[x] += 1
                                else:
                                    combinaison[x] = 1
                if j >= 0:
                    l = ligne[j:i]
                    for k in range(len(l) - 1):
                        if l[k] != l[-1]:
                            x = tuple(sorted([l[k], l[-1]]))
                            if x in combinaison:
                                combinaison[x] += 1
                            else:
                                combinaison[x] = 1

        data_tuples = [(key[0], key[1], value) for key, value in combinaison.items()]
        df = pd.DataFrame(data_tuples, columns=['Person1', 'Person2', 'Occurrences'])
        df = df.sort_values(by='Occurrences', ascending=False)
        df.reset_index(drop=True, inplace=True)
        df = df.head(200)

        return df

    def draw_graph(self, df):
        l = nx.from_pandas_edgelist(
            df,
            source='Person1',
            target='Person2',
            edge_attr='Occurrences',
            edge_weight='Occurrences',
            create_using=nx.Graph()
        )
        node_degree = dict(l.degree)
        nx.set_node_attributes(l, node_degree, "size")
        net = network(notebook=True, width="1000px", height="700px", font_color="white", cdn_ressource="remote")
        net.from_nx(l)
        html = net.generate_html()
        html = html.replace("\n", " ")
        output_html = f"""
                <iframe style="width: 100%; height: 600px; margin:0 auto" name="result" allow="midi; geolocation; microphone; camera; 
                display-capture; encrypted-media;" sandbox="allow-modals allow-forms 
                allow-scripts allow-same-origin allow-popups 
                allow-top-navigation-by-user-activation allow-downloads" allowfullscreen="" 
                allowpaymentrequest="" frameborder="0" srcdoc="{html}"></iframe>"""
        return output_html
