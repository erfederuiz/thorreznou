import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import iqr
from IPython.display import display
import os

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import classification_report

from wordcloud import WordCloud
from PIL import Image


# FUNCION 1
def visualizeME_palettes_or_colors(selection = 'palette', quantity_colors= 8):
    '''
    Function that returns the possible color palettes of the Seaborn library
    ### Parameters (2):
        * selection: `str` by default gives 'palettes', but you can choose colors
        * quantity_colors: `int` by default it returns 8 colors per palette, you can change number of colors you will need. And if you want to see colors, it is not neccesary this parameter.
    ### Return (1):
        * plt.show(): available palettes/ colors with their respective names
    '''
    colors = pd.read_csv('data/seaborn_color_list.csv')

    if selection == 'palette':
        grid = np.vstack((np.linspace(0, 1, quantity_colors), np.linspace(0, 1, quantity_colors)))
        options_colors = colors['PALETTE_COLORS'].dropna().sort_values(key=lambda x: x.str.lower())
        col = 4                             
        pos = 1 
        row = int(len(options_colors)/col)+1 
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
    
        print('If you want any palette reversed, just add "_r" at the end of the palette name')

    elif selection == 'color':
        just_colors = sorted(colors['CSS4_COLORS'].dropna().sort_values(key=lambda x: x.str.lower()))
        col = 4                             
        pos = 1 
        row = int(len(just_colors)/col)+1 
        plt.figure(figsize=(col*3,row))
        
        for i in just_colors:
            plt.subplot(row, col, pos)
            plt.hlines(0,0,5, color = i ,linestyles = 'solid', linewidth = 25)
            plt.axis('off')
            plt.text(0,0.04, i, fontsize = 20)
            pos = pos + 1

    plt.tight_layout()
    return plt.show()


# FUNCION 2
def visualizeME_and_describe_violinbox(dataframe, categ_var, numeric_var, palette= 'tab10', save= True):
    '''
    Function that allows to obtain a more complete graph by merging boxplot and violinplot together with a table of descriptive metrics
    ### Parameters (5):
        * dataframe: `dataframe`  origin table
        * categ_var: `str` categoric variable
        * numeric_var:  `str` numeric variable
        * palette:  `str` by default 'tab10', but you can choose your palette
        * save:  `bool` by default True, the function save the plot and table generated
    '''
    # Generate ViolinBOX graph
    num_cat = len(list(dataframe[categ_var].unique()))
    plt.figure(figsize=(num_cat*1.5,10))
    sns.violinplot(x=categ_var, y=numeric_var, data=dataframe, palette= palette)
    ax = sns.boxplot(x=categ_var, y=numeric_var, data=dataframe,fliersize=0, color='white')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right');
    titulo= numeric_var.upper() + '_vs_' + categ_var.upper()
    plt.title(titulo, fontsize=15);

    # Save graph
    if save == True:
        graph = 'visualizeME_Graphic_violinbox_' + titulo.lower() + '.png'
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
        name = 'visualizeME_table_violinbox_' + titulo.lower() + '.csv'
        table.to_csv(name, header=True)
    
    plt.show()
    display(table)


# FUNCION 3
def visualizeME_and_describe_barplot(dataframe, categ_var, numeric_var, palette='tab10', save = True):
    '''
    Function that allows to obtain a barplot with a table of descriptive metrics
    ### Parameters (5):
        * dataframe: `dataframe`  origin table
        * categ_var: `str` categoric variable
        * numeric_var:  `str` numeric variable
        * palette:  `str` by default 'tab10', but you can choose your palette
        * save:  `bool` by default True, the function save the plot and table generated
    '''
    # Graph
    num_cat = len(list(dataframe[categ_var].value_counts().index[::-1]))
    plt.figure(figsize=(num_cat*1.5,8))
    ax = sns.barplot(x= categ_var,y= numeric_var, data= dataframe, palette= palette, order= dataframe[categ_var].value_counts().index[::-1], ci=None)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right');
    titulo= numeric_var.upper() + ' vs. ' + categ_var.upper()
    plt.title(titulo, fontsize=15)
    
    # Save graph
    if save == True:
        path=os.path.join('visualizeME_Graphic_barplot_' + titulo.lower() + '.' + 'png')
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
        name = 'visualizeME_table_barplot_' + titulo.lower() + '.csv'
        table.to_csv(name, header=True)
    
    plt.show()
    display(table)


# FUNCION 4
def visualizeME_c_matrix(y_true, 
                        y_pred, 
                        title='',
                        categories=[],
                        rotate=False,
                        cmap='',
                        cbar=True,
                        metrics=True,
                        save=True):
    '''
    This function plots a prettier confusion matrix with useful annotations 
       
    and adds a pd.DataFrame with the most common metrics used in classification.
        
    You can use this function with binary or multiclass classification.
    
    The figsize is calculated based on the number of categories.
        
    For both cases:
        * In the main diagonal you get:
            * counts over total of class
            * percentaje over total observations
        * In the rest of the cells you get:
            * counts
            * percentaje over total observations
        * If counts is zero you get an empty cell
           
    In case the function founds binary catergories in your data then a binary
    
    matrix is displayed with the TN, FN, FP, TP tags.
         
    ### Parameters:
    
        y_true -- `array_like`
            True labels.

        y_pred -- `array_like`
            Predictions to compare with true labels.

        title='' -- `str`
            Title to be displayed and used to save file.

        categories=[] -- `list(str)`
            List of names of the classes to be displayed instead of numeric values.

        rotate=False -- `bool`
            Applies a rotation on xticklabels in case they overlap.

        cmap='' -- `Matplotlib colormap name or object, or list of colors` 
            If not provided default is 'Blues'.

        cbar=True -- `bool`
            Whether to draw a colorbar.
            
        metrics=True -- `bool`
            Displays a pd.DataFrame with the most common metrics.

        save=False -- `bool` 
            Saves plot and metrics (if metrics=True) to disk. If title='' default is 'classifier'.
        
    ### Return:
    
        cfm -- `sklearn.metrics.confusion_matrix`
        
        metrics -- `pd.DataFrame`
    '''
    sns.set_theme(style='whitegrid')                
    # Generate confusion matrix
    cfm = confusion_matrix(y_true, y_pred)
    
    # Set tipe of classifier: binary or not binary
    if len(cfm)==2:
        base = len(cfm)
        bin_classifier = True
        annot_kws = {"size": 10*base}
        title_fontsize = 12*base
        axis_fontsize = 10*base
        labels_fontsize = 10*base
        plt.figure(figsize=(4.5*base, 4.5*base))
        
    else:
        base = len(cfm)
        bin_classifier = False    
        annot_kws = {"size": 1.5*base}
        title_fontsize = 3.5*base
        axis_fontsize = 2.5*base
        labels_fontsize = 1.5*base
        plt.figure(figsize=(1.75*base, 1.75*base))
    
    # Calculate auxiliar data   
    cfm_rowsum  = np.sum(cfm, axis=1, keepdims=True)
    
    if bin_classifier:
        cfm_percent = cfm / np.sum(cfm).astype(float)
    else:
        cfm_percent = cfm / cfm_rowsum.astype(float)
    
    # Build empty matrix for labels
    labels = np.zeros_like(cfm).astype(str)
    
    # Iterate labels to write correct annot
    nrows, ncols = cfm.shape
    
    for i in range(nrows):
        for j in range(ncols):
            count = cfm[i, j]
            percent = cfm_percent[i, j]
            
            if i == j:
                sum = cfm_rowsum[i]
                labels[i, j] = f'{count} / {int(sum)}\n{percent:.2%}'
            elif count == 0:
                labels[i, j] = ''
            else:
                labels[i, j] = f'{count}\n{percent:.2%}'
    
    if bin_classifier:
        names = ['TN', 'FP', 'FN', 'TP']
        labels = [name + '\n' + label for name, label in zip(names, labels.flatten())]
        labels = np.asarray(labels).reshape(2, 2)
    
    # Set color map
    if cmap == '':
        cmap = 'Blues'
    else:
        pass
        
    # Generate heatmap
    ax = sns.heatmap(cfm,
                     annot=labels,
                     annot_kws=annot_kws,
                     fmt='',
                     square=True,
                     cmap=cmap,
                     cbar=cbar)
        
    # Define categories position
    cat_position = [(i + .5) for i in range(len(labels))]
    
    # Add label rotation        
    if rotate:
        xdegree=50
        plt.yticks(rotation=0)
    else:
        xdegree=0
        plt.yticks(rotation=0)
    
    # Set title label
    if title == '':
        title = 'TRUE VS. PREDICTED'
    else:
        pass
    
    ax.set_title(f'{title.upper()}', fontsize=title_fontsize)
    
    # Set axis labels
    ax.set_xlabel('PREDICTED LABEL', fontsize=axis_fontsize)
    ax.xaxis.set_label_position('bottom')
    ax.set_ylabel('TRUE LABEL', fontsize=axis_fontsize)
    ax.yaxis.set_label_position('left')
    
    # Set tick labels
    # Define category names
    if categories != []:
        try:
            categories = [label for label in categories]
            ax.set_xticks(cat_position)
            ax.set_xticklabels(categories, fontsize=labels_fontsize, rotation=xdegree)
            ax.xaxis.tick_bottom()
            
            ax.set_yticklabels(categories, fontsize=labels_fontsize)
            ax.yaxis.tick_left()

        except ValueError:
            print('''Impossible to parse categories with number of classes. Ticklabels set to numeric.''')
            
    # Save plot
    if save:
        name = 'visualizeME_cf_matrix_' + title.lower() + '.png'
        path=os.path.join(name + '.' + 'png')
        plt.savefig(path, format='png', dpi=300)

    # Plot
    plt.show()
        
    # Calculate metrics
    if metrics:
        if bin_classifier:
            metrics_df = pd.DataFrame({title: [f'{accuracy_score(y_true, y_pred):.10f}',
                                            f'{precision_score(y_true, y_pred):.10f}',
                                            f'{recall_score(y_true, y_pred):.10f}',
                                            f'{f1_score(y_true, y_pred):.10f}',
                                            f'{roc_auc_score(y_true, y_pred):.10f}']},
                                   index=[['Accuracy: (TP + TN) / TOTAL',
                                           'Precision: TP / (TP + FP)',
                                           'Recall: TP / (TP + FN)',
                                           'F1: harmonic mean (accuracy, recall)',
                                           'ROC AUC']])
        else:
            report = classification_report(y_true, y_pred)
            report = [line.split(' ') for line in report.splitlines()]

            header = [x.upper() for x in report[0] if x!='']

            index = []
            values = []

            for row in report[1:-5]:
                row = [value for value in row if value!='']
                if row!=[]:
                    index.append(row[0].upper())
                    values.append(row[1:])

            index.append('ACCURACY')
            values.append(['-', '-'] + [x for x in report[-3] if x != ''][-2:])
            index.append('MACRO AVG.')
            values.append([x for x in report[-2] if x != ''][-4:])
            index.append('WEIGHTED AVG.')
            values.append([x for x in report[-1] if x != ''][-4:])

            metrics_df = pd.DataFrame(data=values, columns=header, index=index)

        # Plot metrics
        display(metrics_df)
    
    # Save metrics
    if save:
        name = 'visualizeME_cf_matrix_' + title.lower() + '.csv'
        metrics_df.to_csv(name, header=True)

    return cfm, metrics


# FUNCION 5
def visualizeME_FigureWords(dataframe, categ_var, shape= 'seahorse', cmap= 'tab10', contour= 'steelblue', back_color = 'white', height= 18, width = 20, save= True):
    '''
    Function that returns graph of words with different shapes, with the possibility to choose between 'dino', 'heart', 'star', 'seahorse' and 'hashtag'. I hope you like it!
    ### Parameters (9):
        * dataframe: `dataframe` origin table
        * categ_var: `str` categoric variable
        * shape: `str` by default is 'seahorse' shape, but you can choose from this list: 'seahorse', 'dino', 'heart', 'star' and 'hashtag'.
        * cmap: `str` by default is 'tab10', but you can choose your palette of Seaborn. If you want to know which palettes are available you can call visualizeME_colors_palettes() function
        * contour: `str` by default is 'steelblue', but you can choose your favourite color
        * back_color: `str` by default is 'white', but you can choose your background color
        * height: `int` by default is 18, but you can select your preference on height of the figure
        * width:`int` by default is 20, but you can select your preference on width of the figure
        * save: `bool` by default is True in order to save your graph, but if you prefer don't save it, just choose 'False'
    ### Return (1):
        * plt.show(): graph with your figure(by default will be seahorse)
    '''
    # Shape
    while shape not in ['dino', 'heart', 'star', 'seahorse', 'hashtag']:    
        shape = input('Try again, what shape do you want for your figure words graph?\n*Dino\n*Heart\n*Star\n*Seahorse: ').lower()
    if shape == 'seahorse':
        figure = 'data/seahorse_visualizeME.jpg'
    elif shape == 'dino':
        figure = 'data/dino_steg_visualizeME.jpg'
    elif shape == 'heart': 
        figure = 'data/corazon_visualizeME.png'
    elif shape == 'star':
        figure = 'data/estrella-silueta_visualizeME.png'  
    elif shape == 'hashtag':
        figure = 'data/hashtag-silueta_visualizeME.png'
    
    # Words
    words = ' '.join(map(str, dataframe[categ_var]))
    custom_mask = np.array(Image.open(figure))
    wordcloud = WordCloud(background_color=back_color,
                      width=2500,
                      height=2000,
                      max_words=500, 
                      contour_width=0.1, 
                      contour_color= contour, 
                      colormap= cmap,
                      scale =5,mask=custom_mask).generate(words)
    
    plt.figure(1, figsize = (height, width))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

    # Save Graph
    if save == True:
        figure = figure.split('/')[1]
        figure = figure.split('.')[0]
        name = 'visualizeME_Graphic_' + figure + '.png'
        plt.savefig(name)
    
    return plt.show()


# FUNCION 6
def visualizeME_bagel_look_top(dataframe, categ_var, top=0, cmap = 'tab10', circle=True, save=True):
    '''
    Function to generate a bagel graphic where you can select the top categories you want to see, or everyone by default
    ### Parameters (6):
        * dataframe: `dataframe` origin table
        * categ_var: `str` categoric variable
        * top: `int` by default is 0, but you can choose look the top n categories and their weights.
        * cmap: `str` by default is 'tab10', but you can choose your palette of Seaborn. If you want to know which palettes are available you can call visualizeME_colors_palettes() function
        * circle: `bool` by default is True, in orders to seems like a bagel, but you can choose select a pie
        * save: `bool` by default is True in order to save your graph, but if you prefer don't save it, just choose 'False'
    '''
    data_valores= dataframe[categ_var].value_counts()
    # Filter by top categories
    if top != 0:
        data_valores_top= data_valores[(top):]
        p=0
        for i in data_valores_top:
            p=i+p
        data_valores_top=pd.Series([p],index=[f"Resto [{len(data_valores_top.index)}]"])
        data_valores_max= data_valores[:(top)]
        data_valores=pd.concat([data_valores_max, data_valores_top])
    
    # Generate graph
    plt.figure(figsize=(10,10))
    plt.pie(data_valores.values, labels=data_valores.index, textprops={"fontsize":15}, startangle = 60, autopct='%1.2f%%', frame=False, colors= sns.color_palette(cmap))
    p=plt.gcf()
    if circle is True:
        my_circle=plt.Circle((0,0), 0.4, color="w")
        p.gca().add_artist(my_circle)
    titulo= 'DISTRIBUCIÓN DE ' + categ_var.upper()
    plt.title(titulo, fontsize= 20)

    # Save graph
    if save == True:
        path=os.path.join('visualizeME_Graphic_baggel_' + titulo.lower() + '.png')
        plt.savefig(path, format='png', dpi=300)
    
    # Generate table
    values_bagel = pd.DataFrame(dataframe[categ_var].value_counts())
    new_list = []
    for i in list(dataframe[categ_var].value_counts()):
        sumat = sum(list(dataframe[categ_var].value_counts()))
        peso = i/sumat
        new_list.append(peso)
    porcentaj_nums = list(map(lambda x : x * 100, new_list))
    porcentaj_round3 = list(map(lambda x : round(x,2), porcentaj_nums))
    porcentaj = list(map(lambda x : str(x) + '%', porcentaj_round3))
    values_bagel['Pesos(%)'] = porcentaj

    # Save table
    if save == True:
        name = 'visualizeME_table_bagel_' + titulo.lower() + '.csv'
        values_bagel.to_csv(name, header=True)
    
    plt.show()
    return display(values_bagel)


#FUNCION 7
def visualizeME_and_describe_Spidey(dataframe, save= True):
    '''
    This function  generate a polar chart with your numeric variables in order to compare their means. Important!! first scale your numeric variables.
    ### Parameters (2):
        * dataframe: `dataframe` origin table
        * save: `bool` by default is True in order to save your graph, but if you prefer don't save it, just choose 'False'
    '''
    spidey = pd.DataFrame(dataframe.iloc[:,0:-1].mean(), columns=['Means'])

    categories=list(dataframe.iloc[:,0:-1].columns)
    categories+=categories[:1]
    num =len(categories)

    # variables means
    value=list(dataframe.iloc[:,0:-1].mean())
    value+=value[:1]

    loc_label = np.linspace(start=0, stop=2*np.pi, num= num)

    plt.figure(figsize=(10,10))
    ax = plt.subplot(polar=True)
    plt.plot(loc_label, value)
    plt.fill(loc_label, value, 'blue', alpha=0.1)
    # use thetagrids to place labels at the specified angles using degrees
    lines, labels = plt.thetagrids(np.degrees(loc_label), labels=categories)

    # Comienza radar chart arriba y hacia la derecha las variables
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)

    titulo= 'SPIDEY CHART TO COMPARE MEANS OF SCALED NUMERIC VARIABLES'
    plt.title(titulo, y=1.1, fontdict={'fontsize': 18})
    plt.legend(labels=['Mean'],loc=(1, 1))

    # Save graph
    if save == True:
        path=os.path.join('visualizeME_Graphic_' + titulo.lower() + '.png')
        plt.savefig(path, format='png', dpi=300)

    # Save Table
    if save == True:
        name = 'visualizeME_table_' + titulo.lower() + '.csv'
        spidey.to_csv(name, header=True)

    plt.show()
    return display(spidey)



# FUNCION 8 -> La deja subida Natalia

def basic_scatterplot(dataframe,numeric_var1,numeric_var2,title,xlabel,ylabel,palette):
    '''
    This function shows the graph scatterplot.
    ### Parameters(7):
        * dataframe: `dataframe`  origin table
        * numeric_var1: `str` variable
        * numeric_var2: `str` variable
        * title: 'str' graph's title
        * xlabel: Function x title
        * ylabel: Function x title
        * palette: `str` argument to choose color palett
    '''
    plt.figure(figsize=(13,7))
    sns.scatterplot(data = dataframe, x = numeric_var1, y = numeric_var2, palette = palette)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def ej_basic_scatterplot():
    '''
    This function shows an example of the graph
    '''
    dataframe = sns.load_dataset("tips")
    print("SCATTER PLOT")
    basic_scatterplot(dataframe,"tip","total_bill","TIP VS TOTAL PAID","Tip","Total paid","tab10")

ej_basic_scatterplot()


def write_basic_scatterplot():
    '''
    This function shows the graph's code.
    '''
    print("You can use this function -> basic_scatterplot(dataframe,numeric_var1,numeric_var2,title,xlabel,ylabel,palette)")

write_basic_scatterplot()



# STATTER PLOT
# ADVANCE

def advance_scatterplot(dataframe,numeric_var1,numeric_var2,categ_var,title,xlabel,ylabel,palette):
    '''
    This function shows the graph scatterplot.
    ### Parameters(9):
        * dataframe: `dataframe`  origin table
        * numeric_var1: `str` variable
        * numeric_var2: `str` variable
        * categ_var: `str` variable
        * title: 'str' graph's title
        * xlabel: Function x title
        * ylabel: Function x title
        * palette: `str` argument to choose color palett
    '''
    plt.figure(figsize=(13,7))
    sns.scatterplot(data = dataframe, x = numeric_var1, y = numeric_var2, 
                    hue = categ_var, palette = palette,
                    style = categ_var,markers = ["^", "v"])
    plt.legend(bbox_to_anchor=(1, 1), loc=2) ;
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()  


def ej_advance_scatterplot():
    '''
    This function shows an example of the graph
    '''
    dataframe = sns.load_dataset("tips")
    print("SCATTER PLOT")
    advance_scatterplot(dataframe,"tip","total_bill","sex","TIP VS TOTAL PAID","Tip","Total paid","tab10")

ej_advance_scatterplot()


def write_advance_scatterplot():
    '''
    This function shows the graph's code.
    '''
    print("You can use this function -> advance_scatterplot(dataframe,numeric_var1,numeric_var2,categ_var,title,xlabel,ylabel,palette)")

write_advance_scatterplot()


# LINE PLOT
# BASIC

def basic_lineplot(dataframe,numeric_var1,numeric_var2,title,xlabel,ylabel,palette):
    '''
    This function shows the graph lineplot.
    ### Parameters(7):
        * dataframe: `dataframe`  origin table
        * numeric_var1: `str` variable
        * numeric_var2: `str` variable
        * title: 'str' graph's title
        * xlabel: Function x title
        * ylabel: Function x title
        * palette: `str` argument to choose color palett
    '''
    plt.figure(figsize=(13,7))
    sns.lineplot(data = dataframe, x = numeric_var1, y = numeric_var2, palette = palette)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def ej_basic_lineplot():
    '''
    This function shows an example of the graph
    '''
    dataframe = sns.load_dataset("flights")
    print("LINE PLOT")
    basic_lineplot(dataframe,"year","passengers","TOTAL PASSENGERS PER YEAR","Year","Total passengers","tab10")

ej_basic_lineplot()


def write_basic_lineplot():
    '''
    This function shows the graph's code.
    '''
    print("You can use this function -> basic_lineplot(dataframe,numeric_var1,numeric_var2,title,xlabel,ylabel,palette)")
    
write_basic_lineplot()



# LINE PLOT
# ADVANCE

def advance_lineplot(dataframe,numeric_var1,numeric_var2,categ_var,title,xlabel,ylabel,palette):
    '''
    This function shows the graph lineplot.
    ### Parameters(8):
        * dataframe: `dataframe`  origin table
        * numeric_var1: `str` variable
        * numeric_var2: `str` variable
        * categ_var: `str` variable
        * title: 'str' graph's title
        * xlabel: Function x title
        * ylabel: Function x title
        * palette: `str` argument to choose color palett
    '''
    plt.figure(figsize=(13,7))
    sns.lineplot(data = dataframe, x = numeric_var1, y = numeric_var2,
                hue = categ_var, palette = palette)
    plt.legend(bbox_to_anchor=(1, 1), loc=2) ;
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def ej_advance_lineplot():
    '''
    This function shows an example of the graph
    '''
    dataframe = sns.load_dataset("flights")
    print("LINE PLOT")
    advance_lineplot(dataframe,"year","passengers","month","TOTAL PASSENGERS PER YEAR","Year","Total passengers","tab10")

ej_advance_lineplot()


def write_advance_lineplot():
    '''
    This function shows the graph's code.
    '''
    print("You can use this function -> advance_lineplot(dataframe,numeric_var1,numeric_var2,categ_var,title,xlabel,ylabel,palette)")

write_advance_lineplot()



# HISTPLOT
# BASIC
def basic_histplot(dataframe,numeric_var,title,xlabel,palette):
    '''
    This function shows the graph histplot.
    ### Parameters(5):
        * dataframea: `dataframe`  origin table
        * numeric_var: `str` variable
        * title: 'str' graph's title
        * xlabel: Function x title
        * palette: `str` argument to choose color palett
    '''
    plt.figure(figsize=(13,7))
    sns.histplot(data = dataframe, x = numeric_var, palette = palette)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.show()


def ej_basic_histplot():
    '''
    This function shows an example of the graph
    '''
    dataframe = sns.load_dataset("penguins")
    print("HISTPLOT")
    basic_histplot(dataframe,"body_mass_g","PENGUINS BODY MASS", "Body mass","tab10")

ej_basic_histplot()


def write_basic_histplot():
    '''
    This function shows the graph's code.
    '''
    print("You can use this function -> basic_histplot(dataframe,numeric_var,title,xlabel,palette)")

write_basic_histplot()


# HISTPLOT
# ADVANCE
def basic_histplot(dataframe,numeric_var,categ_var,title,xlabel,palette):
    '''
    This function shows the graph histplot.
    ### Parameters(5):
        * dataframea: `dataframe`  origin table
        * x: `str` variable
        * title: 'str' graph's title
        * xlabel: Function x title
        * palette: `str` argument to choose color palett
    '''
    plt.figure(figsize=(13,7))
    sns.histplot(data = dataframe, x = numeric_var, palette = palette, hue = categ_var)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.show()


def ej_basic_histplot():
    '''
    This function shows an example of the graph
    '''
    dataframe = sns.load_dataset("penguins")
    print("HISTPLOT")
    basic_histplot(dataframe,"body_mass_g","sex","PENGUINS BODY MASS", "Body mass","tab10")

ej_basic_histplot()


def write_basic_histplot():
    '''
    This function shows the graph's code.
    '''
    print("You can use this function -> basic_histplot(dataframe,numeric_var,categ_var,title,xlabel,palette)")

write_basic_histplot()


# BARPLOT
# BASIC


def basic_barplot(dataframe,numeric_var,categ_var,title,xlabel,ylabel,palette):
    '''
    This function shows the graph barplot.
    ### Parameters(8):
        * dataframe: `dataframe`  origin table
        * x: `str` variable
        * y: `str` variable
        * tupla_categ_var: `tupla` with categ_var variables
        * title: 'str' graph's title
        * xlabel: Function x title
        * ylabel: Function x title
        * palette: `str` argument to choose color palett
    '''
    plt.figure(figsize=(13,7))
    sns.barplot(data = dataframe, x = numeric_var, y = categ_var, palette = palette,
                ci = None)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def ej_basic_barplot():
    '''
    This function shows an example of the graph
    '''
    dataframe = sns.load_dataset("titanic")
    print("BARPLOT")
    basic_barplot(dataframe,"class","fare","AVERAGE PAID ACCORDING TO SOCIAL CLASS","Social class","Average paid","tab10")

ej_basic_barplot()


def write_basic_barplot():
    '''
    This function shows the graph's code.
    '''
    print("You can use this function -> basic_barplot(dataframe,numeric_var,categ_var,title,xlabel,ylabel,palette)")

write_basic_barplot()


# BARPLOT
# ADVANCE

def advance_barplot(dataframe,numeric_var,categ_var1,categ_var2,title,xlabel,ylabel,palette):
    '''
    This function shows the graph barplot.
    ### Parameters(8):
        * dataframe: `dataframe`  origin table
        * numeric_var: `str` variable
        * categ_var1: `str` variable
        * categ_var2: `str` variable
        * title: 'str' graph's title
        * xlabel: Function x title
        * ylabel: Function x title
        * palette: `str` argument to choose color palett
    '''
    plt.figure(figsize=(13,7))
    sns.barplot(data = dataframe, x = numeric_var, y = categ_var1, 
                hue = categ_var2, palette = palette, ci = None)
    plt.legend(bbox_to_anchor=(1, 1), loc=2) ;
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def ej_advance_barplot():
    '''
    This function shows an example of the graph
    '''
    dataframe = sns.load_dataset("titanic")
    print("BARPLOT")
    advance_barplot(dataframe,"class","fare","sex","AVERAGE PAID ACCORDING TO SOCIAL CLASS","Social class","Average paid","tab10")

ej_advance_barplot()


def write_advance_barplot():
    '''
    This function shows the graph's code.
    '''
    print("You can use this function -> advance_barplot(dataframe,numeric_var,categ_var1,categ_var2,title,xlabel,ylabel,palette)")

write_advance_barplot()


# DENSIDAD
# BASIC

def basic_density(dataframe,numeric_var,title,xlabel,ylabel,palette):
    '''
    This function shows the graph density.
    ### Parameters(6):
        * dataframe: `dataframe`  origin table
        * numeric_var: `str` variable
        * title: 'str' graph's title
        * xlabel: Function x title
        * ylabel: Function x title
        * palette: `str` argument to choose color palett
    '''
    plt.figure(figsize=(13,7))
    sns.kdeplot(data = dataframe, x = numeric_var, palette = palette)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def ej_basic_density():
    '''
    This function shows an example of the graph
    '''
    dataframe = sns.load_dataset("titanic")
    print("DENSITY")
    basic_density(dataframe,"age", "PASSSENGERS AGE", "Age", "Density","tab10")

ej_basic_density()


def write_basic_density():
    '''
    This function shows the graph's code.
    '''
    print("You can use this function -> basic_density(dataframe,numeric_var,title,xlabel,ylabel,palette)")

write_basic_density()


# DENSIDAD
# ADVANCE

def advance_density(dataframe,numeric_var,categ_var,title,xlabel,ylabel,palette):
    '''
    This function shows the graph density.
    ### Parameters(7):
        * dataframe: `dataframe`  origin table
        * numeric_var: `str` variable
        * categ_var: `str` variable
        * title: 'str' graph's title
        * xlabel: Function x title
        * ylabel: Function x title
        * palette: `str` argument to choose color palett
    '''
    plt.figure(figsize=(13,7))
    sns.kdeplot(data = dataframe, x = numeric_var, 
                hue = categ_var, palette = palette, multiple = "stack")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def ej_advance_density():
    '''
    This function shows an example of the graph
    '''
    dataframe = sns.load_dataset("titanic")
    print("DENSITY PLOT")
    advance_density(dataframe,"age", "sex", "PASSSENGERS AGE", "Age", "Density","tab10")

ej_advance_density()


def write_advance_density():
    '''
    This function shows the graph's code.
    '''
    print("You can use this function -> advance_density(dataframe,numeric_var,categ_var,title,xlabel,ylabel,palette")

write_advance_density()


# BOXPLOT
# BASIC

def basic_boxplot(dataframe,categ_var,numeric_var,title,xlabel,ylabel,palette):
    '''
    This function shows the graph boxplot.
    ### Parameters(7):
        * dataframe: `dataframe`  origin table
        * categ_var: `str` variable
        * numeric_var: `str` variable
        * title: 'str' graph's title
        * xlabel: Function x title
        * ylabel: Function x title
        * palette: `str` argument to choose color palett
    '''
    plt.figure(figsize=(13,7))
    sns.boxplot(data = dataframe, x = categ_var, y = numeric_var, palette = palette)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def ej_basic_boxplot():
    '''
    This function shows an example of the graph
    '''
    dataframe = sns.load_dataset("tips")
    print("BOXPLOT")
    basic_boxplot(dataframe,"day","tip","TIP DEPENDING ON THE DAY","Day","Tip","tab10")

ej_basic_boxplot()


def write_basic_boxplot():
    '''
    This function shows the graph's code.
    '''
    print("You can use this function -> basic_boxplot(dataframe,categ_var,numeric_var,title,xlabel,ylabel,palette)")

write_basic_boxplot()



# BOXPLOT
# ADVANCE

def advance_boxplot(dataframe,categ_var1, numeric_var1, categ_var2,title,xlabel,ylabel,palette):
    '''
    This function shows the graph boxplot.
    ### Parameters(8):
        * dataframe: `dataframe`  origin table
        * categ_var1: `str` variable
        * categ_var2: `str` variable
        * numeric_var1: `str` variable
        * title: 'str' graph's title
        * xlabel: Function x title
        * ylabel: Function x title
        * palette: `str` argument to choose color palett
    '''
    plt.figure(figsize=(13,7))
    sns.boxplot(data = dataframe, x = categ_var1, y = numeric_var1,
                palette = palette, hue = categ_var2)
    plt.legend(bbox_to_anchor=(1, 1), loc=2) ;
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def ej_advance_boxplot():
    '''
    This function shows an example of the graph
    '''
    dataframe = sns.load_dataset("tips")
    print("BOXPLOT")
    advance_boxplot(dataframe,"time","tip","smoker","TIP DEPENDING ON THE DAY","Day","Tip","tab10")

ej_advance_boxplot()


def write_advance_boxplot():
    '''
    This function shows the graph's code.
    '''
    print("You can use this function -> advance_boxplot(dataframe,categ_var1,numeric_var1,categ_var2,title,xlabel,ylabel,palette)")

write_advance_boxplot()



# VIOLINPLOT
# BASIC

def basic_violinplot(dataframe,categ_var,numeric_var,title,xlabel,ylabel,palette):
    '''
    This function shows the graph violinplot.
    ### Parameters(6):
        * dataframe: `dataframe`  origin table
        * categ_var: `str` variable
        * numeric_var: `str` variable
        * title: 'str' graph's title
        * xlabel: Function x title
        * ylabel: Function x title
        * palette: `str` argument to choose color palett
    '''
    plt.figure(figsize=(13,7))
    sns.violinplot(data = dataframe, x = categ_var, y = numeric_var, palette = palette)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def ej_basic_violinplot():
    '''
    This function shows an example of the graph
    '''
    dataframe = sns.load_dataset("tips")
    print("VIOLINPLOT")
    basic_violinplot(dataframe,"day","tip","TIP DEPENDING ON THE DAY","Day","Tip","tab10")

ej_basic_violinplot()


def write_basic_violinplot():
    '''
    This function shows the graph's code.
    '''
    print("You can use this function -> basic_violinplot(dataframe,categ_var,numeric_var,title,xlabel,ylabel,palette)")

write_basic_violinplot()



# VIOLINPLOT
# ADVANCE

def advance_violinplot(dataframe,categ_var1,numeric_var,categ_var2,title,xlabel,ylabel,palette):
    '''
    This function shows the graph violinplot.
    ### Parameters(7):
        * dataframe: `dataframe`  origin table
        * categ_var1: `str` variable
        * categ_var2: `str` variable
        * numeric_var: `str` variable
        * title: 'str' graph's title
        * xlabel: Function x title
        * ylabel: Function x title
        * palette: `str` argument to choose color palett
    '''
    plt.figure(figsize=(13,7))
    sns.violinplot(data = dataframe, x = categ_var1, y = numeric_var, palette = palette,hue=categ_var2)
    plt.legend(bbox_to_anchor=(1, 1), loc=2) ;
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def ej_advance_violinplot():
    '''
    This function shows an example of the graph
    '''
    dataframe = sns.load_dataset("tips")
    print("VIOLINPLOT")
    advance_violinplot(dataframe,"day","tip","smoker","TIP DEPENDING ON THE DAY","Day","Tip","tab10")

ej_advance_violinplot()


def write_advance_violinplot():
    '''
    This function shows the graph's code.
    '''
    print("You can use this function -> advance_violinplot(dataframe,categ_var1,y,categ_var2,title,xlabel,ylabel,palette)")

write_advance_violinplot()


# COUNTPLOT
# BASIC

def basic_countplot(dataframe,categ_var,title,xlabel,palette):
    '''
    This function shows the graph countplot.
    ### Parameters(5):
        * dataframe: `dataframe`  origin table
        * categ_var: `str` variable
        * title: 'str' graph's title
        * xlabel: Function x title
        * palette: `str` argument to choose color palett
    '''
    plt.figure(figsize=(13,7))
    sns.countplot(data=dataframe,x=categ_var, palette=palette)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.show()


def ej_basic_countplot():
    '''
    This function shows an example of the graph
    '''
    dataframe = sns.load_dataset("titanic")
    print("COUNTPLOT")
    basic_countplot(dataframe,"class","TITANIC","class","tab10")

ej_basic_countplot()


def write_basic_countplot():
    '''
    This function shows the graph's code.
    '''
    print("You can use this function -> basic_countplot(dataframe,categ_var,title,xlabel,palette)")

write_basic_countplot()


# COUNTPLOT
# ADVANCE

def advance_countplot(dataframe,categ_var1,categ_var2,title,xlabel,palette):
    '''
    This function shows the graph countplot.
    ### Parameters(6):
        * dataframe: `dataframe`  origin table
        * categ_var1: `str` variable
        * categ_var2: `str` variable
        * title: 'str' graph's title
        * xlabel: Function x title
        * palette: `str` argument to choose color palett
    '''
    plt.figure(figsize=(13,7))
    sns.countplot(data=dataframe,x=categ_var1, hue=categ_var2, palette=palette)
    plt.legend(bbox_to_anchor=(1, 1), loc=2) ;
    plt.title(title)
    plt.xlabel(xlabel)
    plt.show()


def ej_advance_countplot():
    '''
    This function shows an example of the graph
    '''
    dataframe = sns.load_dataset("titanic")
    advance_countplot(dataframe,"class","who","TITANIC","class","tab10")

ej_advance_countplot()


def write_advance_countplot():
    '''
    This function shows the graph's code.
    '''
    print("You can use this function -> advance_countplot(dataframe,categ_var1,categ_var2,title,xlabel,palette)")

write_advance_countplot()



# PIE
# BASIC

def basic_pie(dataframe,categ_var,title):
    '''
    This function shows the graph pie.
    ### Parameters(3):
        * dataframe: `dataframe`  origin table
        * x: `str` variable
        * categ_var: `str` variable
    '''
    plt.figure(figsize=(13,7))
    df1 = pd.DataFrame(dataframe[categ_var].value_counts().reset_index())
    plt.pie(data=df1,x=categ_var,labels='index', autopct='%.1f%%')
    plt.title(title)
    plt.show()


def ej_basic_pie():
    '''
    This function shows an example of the graph
    '''
    dataframe = sns.load_dataset("titanic")
    print("PIE")
    basic_pie(dataframe,"sex","SEX")

ej_basic_pie()


def write_basic_pie():
    print("You can use this function -> basic_pie(dataframe,categ_var,title)")
    '''
    This function shows the graph's code.
    '''

write_basic_pie()



def var_question():
    '''
    This function allows to choose how many features the user needs and if they are numeric or categorical values.
    '''
    while True:
        var = input(str('Do you need 1, 2 or 3 features?'))
        if var == "1":
            if var == "1":
                types = input(str('What do you want to cross? 1: a numeric variable, 2: a category variable (Please answer 1 or 2')).lower()
                if types == "1":
                    types = "numeric"
                    return types
                elif types == "2":
                    types = "categoric"
                    return types
        elif var == "2":
            types = input(str('What do you want to cross? 1: a numeric variable with a category variable, 2: a numeric with another numeric, 3: a category with another category (Please answer 1 or 2')).lower()
            if types == "1":
                types = "numeric_categoric"
                return types
            if types == "2":
                types = "numeric_numeric"
                return types
            if types == "3":
                types = "categoric_categoric"
                return types
        elif var == "3":
            types = input(str('What do you want to cross? op.1: one numerical variable with two categories, op.2: two numerical variables with one category (Please answer 1 or 2')).lower()
            if types == "1":
                types = "numeric_categoric_categoric"
                return types
            if types == "2":
                types = "numeric_numeric_categoric"
                return types


def show_graph(types):
    '''
    This function shows, according to whether values are numeric or categorical, what kind of graph the user can choose from.
    ### Parameters (1):
        * types: It specifies whether it is a numeric or categorical kind of feature.
    ### Return (1):
        * graph: Returns the graph
    
    '''
    while True:
        if types == "categoric":
            print("1")
            ej_basic_countplot()
            print("2")
            ej_basic_density()
            print("3")
            ej_basic_pie()
            graph = input('What graph do you want to use? Choose a number').lower() 
            return graph

        elif types == "numeric":
            print("1")
            ej_basic_histplot()
            graph = input('What graph do you want to use? Choose a number').lower() 
            return graph

        elif types == "numeric_categoric":
            print("1")
            ej_basic_barplot()
            print("2")
            ej_basic_boxplot()
            print("3")
            ej_basic_violinplot()
            graph = input('What graph do you want to use? Choose a number').lower() 
            return graph
            
        elif types == "numeric_numeric":
            print("1")
            ej_basic_scatterplot()
            print("2")
            ej_basic_lineplot()
            print("3")
            ej_basic_histplot()
            graph = input('What graph do you want to use? Choose a number').lower() 
            return graph

        elif types == "categoric_categoric":
            print("1")
            ej_basic_countplot()
            graph = input('What graph do you want to use? Choose a number').lower() 
            return graph

        elif types == "numeric_categoric_categoric":
            print("1")
            ej_advance_barplot()
            print("2")
            ej_advance_boxplot()
            print("3")
            ej_advance_violinplot()
            graph = input('What graph do you want to use? Choose a number').lower() 
            return graph


        elif types == "numeric_numeric_categoric":
            print("1")
            ej_advance_scatterplot()
            print("2")
            ej_advance_density()
            print("3")
            ej_advance_lineplot()
            graph = input('What graph do you want to use? Choose a numbero').lower() 
            return graph



def select_graph(types,graph):
    '''
    This function prints the line of code that you will need to create the graph.
    ### Parameters (2):
        * types: It specifies whether it is a numeric or categorical kind of feature.
        * graph: Specify the graph
    '''
    if types == "categoric" and graph == "1":
        write_basic_countplot()
    elif types == "categoric" and graph == "2":
        write_basic_density()
    elif types == "categoric" and graph == "3":
        write_basic_pie()
    
    elif types == "numeric" and graph == "1":
        write_basic_histplot()

    elif types == "numeric_categoric" and graph == "1":
        write_basic_barplot()
    elif types == "numeric_categoric" and graph == "2":
        write_basic_boxplot()
    elif types == "numeric_categoric" and graph == "3":
        write_basic_violinplot()

    elif types == "numeric_numeric" and graph == "1":
        write_advance_scatterplot()
    elif types == "numeric_numeric" and graph == "2":
        write_basic_lineplot()
    elif types == "numeric_numeric" and graph == "3":
        write_basic_histplot()

    elif types == "categoric_categoric" and graph == "1":
        write_basic_countplot()

    elif types == "numeric_categoric_categoric" and graph == "1":
        write_advance_barplot()
    elif types == "numeric_categoric_categoric" and graph == "2":
        write_advance_boxplot()
    elif types == "numeric_categoric_categoric" and graph == "3":
        write_advance_violinplot()

    elif types == "numeric_numeric_categoric" and graph == "1":
        write_advance_scatterplot()
    elif types == "numeric_numeric_categoric" and graph == "2":
        write_advance_density()
    elif types == "numeric_numeric_categoric" and graph == "3":
        write_advance_lineplot()

    else:
        print("There is no function available for those parameters.")


def visualize_select_graph():
    '''
    This function unifies all the previous ones:
        * var_question() - This function allows to choose how many features the user needs and if they are numeric or categorical values.
        * show_graph(types) - This function shows, according to whether values are numeric or categorical, what kind of graph the user can choose from.
        * select_graph(types,graph) - This function prints the line of code that you will need to create the graph.
        ### Parameters (2):
        * types: It specifies whether it is a numeric or categorical kind of feature.
        * graph: Specify the graph
    '''
    types = var_question()
    graph = show_graph(types)
    select_graph(types,graph)

visualize_select_graph()





# FUNCION 9
def visualizeME_scores_models(y_true,models,bin_multi_classifier,vis_pallete='tab10',save = True):
    ''' 
    This is a function that allows you to quickly identify the best metrics from your Machine Learning models whether is binary or non binary target
    ### Parameters(4):
        * y_true: -- `array_like` 
            True labels.
        * models: 
            here you should put a diccionary that contains for keys, the model names (as a str) and for values their predictions to compare with true labels.
             This will help to join all the metrics in one data frame
             EXAMPLE: {'Linear Regression':preds_multi,'KNN Classifier':preds_multi2}
        * vis_pallete:
            Seaborn color pallete we have selected before
        * bin_multi_classifier:
            True if it is a binomial model (0,1) and  False if it is multicategory
        * save=False -- `bool` 
            Saves plot and metrics (if metrics=True) to disk. If title='' default is 'classifier'.
    '''
    # extracting values and keys from models dictionary:
    df = pd.DataFrame()
    keys_modelo = list(models.keys())
    values_modelo = list(models.values())   
    
    #If we have selected a binomial classifier then:
    if bin_multi_classifier == True:
        #Elaborating a Data Frame that will give you 5 metrics from you ML Model:
        for i,j in enumerate(models):
            metrics_df = pd.DataFrame({str(j): [float(f'{accuracy_score(y_true, values_modelo[i]):.10f}'),
                                            float(f'{precision_score(y_true, values_modelo[i]):.10f}'),
                                            float(f'{recall_score(y_true, values_modelo[i]):.10f}'),
                                            float(f'{f1_score(y_true, values_modelo[i]):.10f}'),
                                            float(f'{roc_auc_score(y_true, values_modelo[i]):.10f}')]},
                                index=[['Accuracy: (TP + TN) / TOTAL',
                                           'Precision: TP / (TP + FP)',
                                           'Recall: TP / (TP + FN)',
                                           'F1: harmonic mean (accuracy, recall)',
                                           'ROC AUC']])
            # Here we are joining all the dataframe metrics from all the models we have selected:
            if i == 0:
                df=metrics_df
            else:
                df = df.join(other=metrics_df)

        #FINDING METRICS
        for y in range(len(df.index)):
            #Finding the best Accuracy:
            if y == 0:
                for i,j in enumerate(df.iloc[y]):
                    if j == float(df.iloc[y].max()):
                        best_acc = df.columns[i]  

                # Setting a bar plot to see the best Accuracy:
                num=len(df.columns)
                plt.figure(figsize=(num*1.5,5))
                graf1= sns.barplot(x=df.columns,y=df.iloc[y],palette=vis_pallete)
                graf1.set_xlabel('Metrics from Models',fontsize = 12)
                graf1.set_title('Accuracy',fontsize=15,color='blue')
                for i in graf1.patches:
                    graf1.annotate(round(i.get_height(),3),(i.get_x() + i.get_width() / 2, i.get_height()),
                        ha='center',va='baseline',fontsize=12,color='black',
                            xytext=(0, 1),textcoords='offset points')
                print('The best Accuracy value is '   + str(df.iloc[y].max())+' from Model: '+ str(best_acc))

                # Saving Bar plot
                if save == True:
                    path=os.path.join('visualizeME_Accuracy_barplot' + '.' + 'png')
                    plt.savefig(path, format='png', dpi=300)
                plt.show()
            
            #Finding the best Precision:
            elif y == 1:
                for i,j in enumerate(df.iloc[y]):
                    if j == float(df.iloc[y].max()):
                        best_prec = df.columns[i]    
            

                # Setting a bar plot to see the best Precision:
                num=len(df.columns)
                plt.figure(figsize=(num*1.5,5))
                graf2= sns.barplot(x=df.columns,y=df.iloc[y],palette=vis_pallete)
                graf2.set_xlabel('Metrics from Models',fontsize = 12)
                graf2.set_title('Precision',fontsize=15,color='blue')
                for i in graf2.patches:
                    graf2.annotate(round(i.get_height(),3),(i.get_x() + i.get_width() / 2, i.get_height()),
                        ha='center',va='baseline',fontsize=12,color='black',
                            xytext=(0, 1),textcoords='offset points')
                print('The best Precision value is '   + str(df.iloc[y].max())+' from Model: '+ best_prec)
                
                
                # Saving Bar plot
                if save == True:
                    name = 'visualizeME_Precision_barplot' 
                    path=os.path.join(name + '.' + 'jpg')
                    plt.savefig(path, format='jpg', dpi=300)
                plt.show()

            #Finding the best Recall:
            elif y == 2:
                for i,j in enumerate(df.iloc[y]):
                    if j == float(df.iloc[y].max()):
                        best_rec = df.columns[i]    

                
                # Setting a bar plot to see the best Recall:
                num=len(df.columns)
                plt.figure(figsize=(num*1.5,5))
                graf3= sns.barplot(x=df.columns,y=df.iloc[y],palette=vis_pallete)
                graf3.set_xlabel('Metrics from Models',fontsize = 12)
                graf3.set_title('Recall',fontsize=15,color='blue')
                for i in graf3.patches:
                    graf3.annotate(round(i.get_height(),3),(i.get_x() + i.get_width() / 2, i.get_height()),
                        ha='center',va='baseline',fontsize=12,color='black',
                            xytext=(0, 1),textcoords='offset points')
                print('The best Recall value is '   + str(df.iloc[y].max())+' from Model: '+ best_rec)
                
                # Saving Bar plot
                if save == True:
                    name = 'visualizeME_Recall_barplot'
                    path=os.path.join(name + '.' + 'jpg')
                    plt.savefig(path, format='jpg', dpi=300)
                plt.show()
            
            #Finding the best F1 score:
            elif y == 3:
                for i,j in enumerate(df.iloc[y]):
                    if j == float(df.iloc[y].max()):
                        best_f1 = df.columns[i]    
            

                # Setting a bar plot to see the best F1 Score:
                num=len(df.columns)
                plt.figure(figsize=(num*1.5,5))
                graf4= sns.barplot(x=df.columns,y=df.iloc[y],palette=vis_pallete)
                graf4.set_xlabel('Metrics from Models',fontsize = 12)
                graf4.set_title('F1 Scores',fontsize=15,color='blue')
                for i in graf4.patches:
                    graf4.annotate(round(i.get_height(),3),(i.get_x() + i.get_width() / 2, i.get_height()),
                        ha='center',va='baseline',fontsize=12,color='black',
                            xytext=(0, 1),textcoords='offset points')
                print('The best F1 Score value is '   + str(df.iloc[y].max())+' from Model: '+ best_f1)
                
                # Saving Bar plot
                if save:
                    name = 'visualizeME_F1Score_barplot' + '.png'
                    path=os.path.join(name + '.' + 'png')
                    plt.savefig(path, format='png', dpi=300)
                plt.show()
            
            #Finding the best ROC:
            elif y == 4:
                for i,j in enumerate(df.iloc[y]):
                    if j == float(df.iloc[y].max()):
                        best_roc = df.columns[i]    
            
                # Setting a bar plot to see the best ROC AUC Score:
                num=len(df.columns)
                plt.figure(figsize=(num*1.5,5))
                graf5= sns.barplot(x=df.columns,y=df.iloc[y],palette= vis_pallete)
                graf5.set_xlabel('Metrics from Models',fontsize = 12)
                graf5.set_title('ROC / AUC',fontsize=15,color='blue')
                for i in graf5.patches:
                    graf5.annotate(round(i.get_height(),3),(i.get_x() + i.get_width() / 2, i.get_height()),
                        ha='center',va='baseline',fontsize=12,color='black',
                            xytext=(0, 1),textcoords='offset points')
                print('The best ROC / AUC value is '   + str(df.iloc[y].max())+' from Model: '+ best_prec)
                
                # Saving Bar plot
                if save == True:
                    name = 'visualizeME_ROC_AUC_barplot' + '.png'
                    #path=os.path.join(name + '.' + 'png')
                    plt.savefig(name,format='png', dpi=300)
                plt.show()


    # bin_multi_classifier is False ->>>>> Multicategorical:
    else:
        for i,j in enumerate(models):
            report = classification_report(y_multi,values_modelo[i])
            report = [line.split(' ') for line in report.splitlines()]

            header = [x.upper()+' : '+keys_modelo[i] for x in report[0] if x!='']

            index = []
            values = []

            for row in report[1:-5]:
                row = [value for value in row if value!='']
                if row!=[]:
                    index.append(row[0].upper())
                    values.append(row[1:])

            index.append('ACCURACY')
            values.append(['-', '-'] + [x for x in report[-3] if x != ''][-2:])
            index.append('MACRO AVG.')
            values.append([x for x in report[-2] if x != ''][-4:])
            index.append('WEIGHTED AVG.')
            values.append([x for x in report[-1] if x != ''][-4:])

            metrics_df = pd.DataFrame(data=values, columns=header, index=index)
            if i == 0:
                df=metrics_df
            elif i == 1:
                df = df.join(other=metrics_df)
            else:
                df = df.join(other=metrics_df,) 
        
        #Creating a Data Frame with the Accuracy metrics from all the models:
        acc_list = []
        
        for i in range(2,len(df.loc['ACCURACY']),4):
                acc_list.append(float(df.loc['ACCURACY'][i]))
        acc_dic={}
        for j,k in enumerate(models):
            acc_dic[k] = acc_list[j]
        
        acc = pd.DataFrame(data= acc_dic,index=keys_modelo)
    
        # Setting a BarPlot to show the Best model Accuracy:
        num=len(acc.index)
        plt.figure(figsize=(num*1.9,4))
        graf1= sns.barplot(data=acc,palette=vis_pallete)
        graf1.set_xlabel('ML Models - Results',fontsize = 12)
        graf1.set_title('Accuracy',fontsize=15,color='blue')
        for i in graf1.patches:
            graf1.annotate(round(i.get_height(),3),(i.get_x() + i.get_width() / 2, i.get_height()),
                        ha='center',va='baseline',fontsize=12,color='black',
                            xytext=(0, 1),textcoords='offset points')
        if save:
            name = 'visualizeME_Accuracy_barplot' + '.png'
            path=os.path.join(name + '.' + 'png')
            plt.savefig(path, format='png', dpi=300)
        plt.show()
    
    # Saving Metrics
    if save == True:
        name = 'visualizeME_df_ML_metrics' + '.csv'
        df.to_csv(name, header=True)


