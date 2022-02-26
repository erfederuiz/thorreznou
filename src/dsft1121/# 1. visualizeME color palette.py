# 1. visualizeME color palette

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


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