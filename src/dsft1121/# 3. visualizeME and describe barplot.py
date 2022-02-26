# 3. visualizeME and describe barplot

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import iqr
from IPython.display import display
import os

def visualizeME_and_describe_barplot(dataframe, categ_var, numeric_var):
    '''
    Function that allows to obtain a barplot with a table of descriptive metrics
    Parameters (3):
        dataframe: origin table
        categ_var: categoric variable
        numeric_var: numeric variable
    '''
    # Graph
    plt.figure(figsize=(16,10))
    ax = sns.barplot(x= categ_var,y= numeric_var, data= dataframe, order= dataframe[categ_var].value_counts().index[::-1], ci=None)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right');
    titulo= numeric_var + ' vs. ' + categ_var
    plt.title(titulo)
    
    # Save graph
    path=os.path.join(titulo + '.' + 'png')
    plt.savefig(path, format='png', dpi=300)
    
    # Metrics table
    eti_media= 'Mean ' + numeric_var
    eti_desv = 'Standard Deviation ' + numeric_var
    cabeceras= ['Metrics',]
    fila1 = ['Number of records',]
    fila2 = [eti_media,]
    fila3 = [eti_desv,]
    d = [fila1, fila2, fila3]
    for i in list(dataframe[categ_var].value_counts().index[::-1]):
        cabeceras.append(i)
        total = str(int(dataframe[dataframe[categ_var].isin([i])][[numeric_var]].count()))
        fila1.append(total)
        media = round(float(dataframe[dataframe[categ_var].isin([i])][[numeric_var]].mean()), 2)
        fila2.append(media)
        desv = round(float(dataframe[dataframe[categ_var].isin([i])][[numeric_var]].std()), 2)
        fila3.append(desv)
    table = pd.DataFrame(d, columns=cabeceras)
    table = table.set_index('Metrics')
    
    # Save table
    name = 'visualizeME_table_barplot_' + titulo + '.csv'
    table.to_csv(name, header=True)
    
    plt.show()
    display(table)