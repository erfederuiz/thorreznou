# 5. visualizeME_FigureWords
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from wordcloud import WordCloud
from PIL import Image


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
