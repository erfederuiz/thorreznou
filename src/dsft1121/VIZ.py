import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import iqr
from IPython.display import display
import os


def visualizeME_colors_palettes(quantity_colors= 8):
    '''
    Function that returns the possible color palettes of the Seaborn library
    Parameters (1):
        quantity_colors: number of colors you need, by default it returns 8 colors per palette
    Return (1):
        plt.show(): available color palettes with their respective names
    '''
    colors = pd.read_csv('seaborn_color_list.csv')
    options_colors = colors['PALETTE_COLORS'].dropna().sort_values(key=lambda x: x.str.lower())
    grid = np.vstack((np.linspace(0, 1, quantity_colors), np.linspace(0, 1, quantity_colors)))
    col = 5                       
    row = int(len(options_colors)/col)+1   
    pos = 1  

    plt.figure(figsize=(col*4,row))
    for i in options_colors:
        if '_r' in i:
            pass
        else:
            plt.subplot(row, col, pos)
            plt.imshow(grid, cmap = i, aspect='auto')
            plt.axis('off')
            plt.title(i, loc = 'center', fontsize = 20)
            pos = pos + 1
    plt.tight_layout()
    return plt.show() 


def visualizeME_and_describe_violinbox(dataframe, categ_var, numeric_var, save=True):
    '''
    Function that allows to obtain a more complete graph by merging boxplot and violinplot together with a table of descriptive metrics
    Parameters (4):
        dataframe: origin table
        categ_var: categoric variable
        numeric_var: numeric variable
        save: by default True, the function save the plot and table generated
    '''
    # Generate ViolinBOX graph
    num_cat = len(list(dataframe[categ_var].unique()))
    plt.figure(figsize=(num_cat*1.5,8))
    sns.violinplot(x=categ_var, y=numeric_var, data=dataframe, palette='rainbow')
    ax = sns.boxplot(x=categ_var, y=numeric_var, data=dataframe,fliersize=0, color='white')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right');
    titulo= numeric_var.upper() + '_vs_' + categ_var.upper()
    plt.title(titulo, fontsize=15);

    # Save graph
    if save == True:
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
    if save == True:
        name = 'visualizeME_table_violinbox_' + titulo + '.csv'
        table.to_csv(name, header=True)
    
    plt.show()
    display(table)


def visualizeME_and_describe_barplot(dataframe, categ_var, numeric_var, save = True):
    '''
    Function that allows to obtain a barplot with a table of descriptive metrics
    Parameters (4):
        dataframe: origin table
        categ_var: categoric variable
        numeric_var: numeric variable
        save: by default True, the function save the plot and table generated
    '''
    # Graph
    num_cat = len(list(dataframe[categ_var].value_counts().index[::-1]))
    plt.figure(figsize=(num_cat*1.5,8))
    ax = sns.barplot(x= categ_var,y= numeric_var, data= dataframe, order= dataframe[categ_var].value_counts().index[::-1], ci=None)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right');
    titulo= numeric_var.upper() + ' vs. ' + categ_var.upper()
    plt.title(titulo, fontsize=15)
    
    # Save graph
    if save == True:
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
    if save == True:
        name = 'visualizeME_table_barplot_' + titulo + '.csv'
        table.to_csv(name, header=True)
    
    plt.show()
    display(table)

