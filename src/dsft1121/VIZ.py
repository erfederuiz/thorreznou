import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline



# STATTER PLOT
# BASIC

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