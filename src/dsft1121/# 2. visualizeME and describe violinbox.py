# 2. visualizeME and describe violinbox

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import iqr
from IPython.display import display


def visualizeME_and_describe_violinbox(dataframe, categ_var, numeric_var):
    '''
    Function that allows to obtain a more complete graph by merging boxplot and violinplot together with a table of descriptive metrics
    Parameters (3):
        dataframe: origin table
        categ_var: categoric variable
        numeric_var: numeric variable
    '''
    # Generate ViolinBOX graph
    plt.figure(figsize=(16,10))
    sns.violinplot(x=categ_var, y=numeric_var, data=dataframe, palette='rainbow')
    ax = sns.boxplot(x=categ_var, y=numeric_var, data=dataframe,fliersize=0, color='white')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right');
    titulo= numeric_var + '_&_' + categ_var
    plt.title(titulo);

    # Save graph
    graph = 'visualizeME_Graphic_violinbox_' + titulo + '.png'
    plt.savefig(graph)

    # Metrics table
    cabeceras= ['Metrics',]
    fila1 = ['Upper limit',]
    fila2 = ['Q3',]
    fila3 = ['Median',]
    fila4 = ['Q1',]
    fila5 = ['Lower limit',]  
    iqr_ = iqr(dataframe[numeric_var], nan_policy='omit')
    d = [ fila1, fila2, fila3, fila4, fila5]
    for i in sorted(list(dataframe[categ_var].unique())):
        cabeceras.append(i)
        mediana = round(float(dataframe[dataframe[categ_var].isin([i])][[numeric_var]].median()), 2)
        fila3.append(mediana)
        q1 = round(np.nanpercentile(dataframe[dataframe[categ_var].isin([i])][[numeric_var]], 25), 2)
        fila4.append(q1)
        q3 = round(np.nanpercentile(dataframe[dataframe[categ_var].isin([i])][[numeric_var]], 75), 2)
        fila2.append(q3)
        th1 = round(q1 - iqr_*1.5, 2)
        fila5.append(th1)
        th2 = round(q3 + iqr_*1.5, 2)
        fila1.append(th2)
    table = pd.DataFrame(d, columns=cabeceras)
    table = table.set_index('Metrics')
    
    # Save table
    name = 'visualizeME_table_violinbox_' + titulo + '.csv'
    table.to_csv(name, header=True)
    
    plt.show()
    display(table)