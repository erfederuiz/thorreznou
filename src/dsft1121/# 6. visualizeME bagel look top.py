# 6. visualizeME bagel look top

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