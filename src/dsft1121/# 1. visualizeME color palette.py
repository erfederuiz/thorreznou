# 1. visualizeME palettes or colors

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


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