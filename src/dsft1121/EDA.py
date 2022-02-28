'''
Function that calculates the mathematical expectation of a random variable ('column_name_groups').
Definition: Mathematical expectation, also known as the expected value, is the the sum or integral of all possible values of a random variable, 
or any given function of it, multiplied by the respective probabilities of the values of the variable.
'''
#Libraries:
import pandas as pd
#
#Body of the function:
#
def math_expect(data, column_name_groups= None, column_name_success= None, n_top= 100):
    '''
    Arguments:
    data --> df
    column_name_groups --> name of column with the groups (random variable) to calculate the math expect.
    column_name_success --> name of column that sets the success measurement. ie: daily income or scoring from movies reviews.
    n_top --> default 100, number of occurrencies within the top shortlist. ie: top 100, top 1000 or top 10.
    '''
    #AGRUPAMOS LAS STARTUPS POR INDUSTRIA Y CALCULAMOS EL TOTAL QUE TENEMOS
    len_data = len(data)

    #CREAMOS UN DATAFRAME DEL AGRUPAMIENTO PARA TRABAJAR CON EL 
    groups = pd.DataFrame(data.groupby(column_name_groups)[column_name_groups].count())
    groups.rename(columns={column_name_groups:'groups'}, inplace=True)
    groups.reset_index(inplace=True)

    #ANADIMOS UNA COLUMNA CON EL % DE STARTUPS POR INDUSTRIA COMPARADO CON EL TOTAL
    groups['% per group'] = (groups['groups'] / len_data) * 100

    #ANADIMOS UNA COLUMNA CON EL NUMERO DE STARTUPS POR INDUSTRIA QUE DEBERIA HABER ENTRE LAS 1000 CON MAYOR FUNDING
    groups['expected_in_top'] = (groups['% per group'] * 10)

    #ORDENAMOS POR EL TOTAL FUNDING Y SELECCIONAMOS LAS 100 PRIMERAS
    data.sort_values(by=column_name_success, ascending=False, inplace=True)
    top = data[:n_top]

    #AGRUPAMOS ESAS 100 STARTUPS POR SECTOR Y CONTAMOS CUANTAS HAY DE CADA SECTOR Y ORDENAMOS EN UN DATAFRAME
    top = top.groupby(column_name_groups)[column_name_groups].count()
    top.sort_values(ascending=False, inplace=True)
    top = pd.DataFrame(data=top)
    top.rename(columns={column_name_groups:'actual_top'}, inplace=True)
    top.reset_index(inplace=True)

    #UNIMOS ESE DATAFRAME DE TOP1000 CON EL PRIMER DATAFRAME DEL GRUPO POR SECTOR
    #HACEMOS UN LEFT JOIN Y RELLENAMOS LOS NAN CON 0
    join_group_top = groups.merge(top, how='left', on=column_name_groups)
    join_group_top['actual_top'] = join_group_top['actual_top'].fillna(0)

    #FINALMENTE, CREAMOS UNA COLUMNA QUE CALCULA LA ESPERANZA MATEMATICA
    #ESPERANZA MATEMATICA = (STARTUPS EN EL TOP1000 - LAS QUE DEBERIA HABER EN EL TOP1000) / ESAS QUE DEBERIA HABER EN EL TOP1000 * 100
    join_group_top['math_expectation'] = ((join_group_top['actual_top'] - join_group_top['expected_in_top']) / join_group_top['expected_in_top'])*100
    join_group_top.sort_values(by='math_expectation', ascending=False, inplace=True)

    #YA TENEMOS NUESTRA ESPERANZA MATEMATICA POR SECTOR

    #CREAMOS UN DICCIONARIO EN EL QUE JUNTAREMOS LOS SECTORES CON SUS ESPERANZAS MATEMATICAS
    dicc = {}
    for i, j in zip(join_group_top[column_name_groups], join_group_top['math_expectation']):
        dicc[i] = j

    #MAPEAMOS LOS SECTORES CON SUS ESPERANZAS MATEMATICAS EN LA COLUMNA 'math_expectation_market
    data['math_expectation_market'] = data[column_name_groups].map(dicc) 
    data.sort_values(by='math_expectation_market', ascending=False, inplace=True)

    return data

'''
Function that summarizes data in a new dataframe from an originally given dataframe
'''
#Libraries:
import pandas as pd
#
#Body of the function:
#
def overview(df):
    '''
    Function that summarizes data in a new dataframe from an originally given dataframe.
    Arguments: 
    df --> a DataFrame you are working with
    Output: a new DataFrame with valuable information from the original DataFrame you are working on
    '''
    #Column names
    cols = pd.DataFrame(df.columns.values,columns=['column names'])
    
    #Column types
    types = pd.DataFrame(df.dtypes.values, columns=["type of data"])

    #NAN total
    total_nan = df.isna().sum()
    total_nan = pd.DataFrame(total_nan.values, columns = ["nan total"])

    #NAN %
    percent_nan = round(df.isna().sum()*100/len(df),2)
    percent_nan_df = pd.DataFrame(percent_nan.values, columns = ["nan %"])

    # Sacamos los MISSINGS
    percent_missing = round(df.isnull().sum()*100/len(df),2)
    percent_missing_df = pd.DataFrame(percent_missing.values, columns = ["missing %"])
 
    #Unique values
    unique = pd.DataFrame(df.nunique().values, columns = ["unique values"])
    
    #Cardinality
    percent_cardin = round(unique["unique values"]*100/len(df),2)
    percent_cardin_df = pd.DataFrame(percent_cardin.values, columns = ["cardin %"])

    #We concat the DFs
    concatenate = pd.concat([cols, types, total_nan, percent_nan_df, percent_missing_df, unique, percent_cardin_df], axis=1)
    concatenate.set_index('column names', drop=True, inplace=True)

    #We add the results from describe() to the df
    df_describe = pd.DataFrame(df.describe(), columns=df.columns)
    df_concat = pd.concat([concatenate.T, df_describe])

    return df_concat

'''
Function that removes outliers from a given dataframe, specifying the column.
min and max percentiles are passed as parameters to define the line between outliers and non-outliers.
'''
#Libraries:
import pandas as pd
#
#Body of the function:
#
def outlier_removal(data, column_name= None, p_max=None, p_min=None):
    '''
    Function that removes outliers from a given dataframe, specifying the column.
    min and max percentiles are passed as parameters to define the line between outliers and non-outliers.
    Arguments: 
    data --> a DataFrame you are working with, 
    column_name --> name of the column to detect and remove outliers,
    p_max --> above this percentile we start removing outliers, 
    p_min --> below this percentile we start removing outliers
    Output: a new DataFrame without the data below the p_min and above the p_max
    '''
    q_max = p_max / 100
    q_min = p_min / 100
    q_max_column = data[column_name].quantile(q_max)
    q_min_column = data[column_name].quantile(q_min)

    data = data[(data[column_name] >= q_min_column) & (data[column_name] < q_max_column)]
    return data 