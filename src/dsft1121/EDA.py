#Libraries:
import pandas as pd
import datetime as dt

def math_expect(data, column_name_groups= None, column_name_success= None, n_top= 100):
    '''
    Function that calculates the mathematical expectation of a random variable ('column_name_groups').
    Definition: Mathematical expectation, also known as the expected value, is the the sum or integral of all possible values of a random variable, 
    or any given function of it, multiplied by the respective probabilities of the values of the variable.
    
    Keyword arguments
        - data (Pandas DataFrame type)-- is the given dataframe

        - column_name_groups (string)-- name of column with the groups (random variable) to calculate the math expect

        - column_name_success (string)-- name of column that sets the success measurement. ie: daily income or scoring from movies reviews
        
        - n_top (integer)-- default 100, number of occurrencies within the top shortlist. ie: top 100, top 1000 or top 10

    Return: the DataFrame used as input containing a new column named 'math_expectation' with the resulting values
    '''
    len_data = len(data)

    #WE GROUP BY THE COLUMN WITH THE CHOSEN RANDOM VARIABLE
    groups = pd.DataFrame(data.groupby(column_name_groups)[column_name_groups].count())
    groups.rename(columns={column_name_groups:'groups'}, inplace=True)
    groups.reset_index(inplace=True)

    #WE ADD A COLUMN WITH THE DISTRIBUTION OF EACH GROUP IN %
    groups['% per group'] = (groups['groups'] / len_data) * 100

    #WE CREATE A COLUMN TO SEE THE EXPECTED NUMBER PER GROUP IN THE TOP LIST
    groups['expected_in_top'] = (groups['% per group'] * 10)

    #WE SORT BY THE SUCCESS VARIABLE UNTIL THE N_TOP FIGURE. IE: TOP 10, TOP 100, TOP 1000, ETC
    data.sort_values(by=column_name_success, ascending=False, inplace=True)
    top = data[:n_top]

    #WE GROUP BY AND CALCULATE THE ACTUAL NUMBER OF EACH GROUP VARIABLE IN THE TOP LIST
    top = top.groupby(column_name_groups)[column_name_groups].count()
    top.sort_values(ascending=False, inplace=True)
    top = pd.DataFrame(data=top)
    top.rename(columns={column_name_groups:'actual_top'}, inplace=True)
    top.reset_index(inplace=True)

    #WE MERGE THE TWO DATAFRAMES
    #WE REPLACE THE NAN VALUES FOR ZEROS
    join_group_top = groups.merge(top, how='left', on=column_name_groups)
    join_group_top['actual_top'] = join_group_top['actual_top'].fillna(0)

    #WE CREATE THE MATH_EXPECTATION COLUMN WITH ITS FORMULA AND SORT IT OUT BY THAT COLUMN
    join_group_top['math_expectation'] = ((join_group_top['actual_top'] - join_group_top['expected_in_top']) / join_group_top['expected_in_top'])*100
    join_group_top.sort_values(by='math_expectation', ascending=False, inplace=True)

    #WE CREATE A DICTIONARY WITH THE GROUP NAMES AND MATH_EXPECTATION VALUES
    dicc = {}
    for i, j in zip(join_group_top[column_name_groups], join_group_top['math_expectation']):
        dicc[i] = j

    #WE MAP THE RESULTS AND SORT BY MATH_EXPECTATION COLUMN
    data['math_expectation'] = data[column_name_groups].map(dicc) 
    data.sort_values(by='math_expectation', ascending=False, inplace=True)

    return data

def overview(df):
    '''
    Function that summarizes data in a new dataframe from an originally given dataframe.
  
    Keyword arguments
        - data (Pandas DataFrame type)-- is the given dataframe

    Return: a new DataFrame with valuable information from the original DataFrame you are working on
    '''
    #COLUMN NAMES
    cols = pd.DataFrame(df.columns.values,columns=['column names'])
    
    #COLUMN TYPES
    types = pd.DataFrame(df.dtypes.values, columns=["type of data"])

    #NAN TOTAL
    total_nan = df.isna().sum()
    total_nan = pd.DataFrame(total_nan.values, columns = ["nan total"])

    #NAN %
    percent_nan = round(df.isna().sum()*100/len(df),2)
    percent_nan_df = pd.DataFrame(percent_nan.values, columns = ["nan %"])

    #MISSINGS %
    percent_missing = round(df.isnull().sum()*100/len(df),2)
    percent_missing_df = pd.DataFrame(percent_missing.values, columns = ["missing %"])
 
    #UNIQUE VALUES
    unique = pd.DataFrame(df.nunique().values, columns = ["unique values"])
    
    #CARDINALITY
    percent_cardin = round(unique["unique values"]*100/len(df),2)
    percent_cardin_df = pd.DataFrame(percent_cardin.values, columns = ["cardin %"])

    #WE CONCAT THE DATAFRAMES
    concatenate = pd.concat([cols, types, total_nan, percent_nan_df, percent_missing_df, unique, percent_cardin_df], axis=1)
    concatenate.set_index('column names', drop=True, inplace=True)

    #WE ADD THE DESCRIBE() FUNCTION RESULTS AND CONCATENATE THEM TO THE PREVIOUS DATAFRAME
    df_describe = pd.DataFrame(df.describe(), columns=df.columns)
    df_concat = pd.concat([concatenate.T, df_describe])

    return df_concat

def outlier_removal(data, column_name= None, p_max=None, p_min=None):
    '''
    Function that removes outliers from a given dataframe, specifying the column.
    min and max percentiles are passed as parameters to define the line between outliers and non-outliers.

    Keyword arguments
        - data (Pandas DataFrame type)-- is the given dataframe

        - column_name (string)-- the chosen column to detect and remove outliers from

        - p_max (integer)-- above this percentile we start removing outliers

        - p_min (integer)-- below this percentile we start removing outliers

    Return: a new DataFrame without the data below the p_min and above the p_max
    '''
    q_max = p_max / 100
    q_min = p_min / 100
    q_max_column = data[column_name].quantile(q_max)
    q_min_column = data[column_name].quantile(q_min)

    data = data[(data[column_name] >= q_min_column) & (data[column_name] < q_max_column)]
    return data 

def datetime_into_columns(df, column, weekday = False, hour_minutes = False, from_type = 'object'):
    """ 
    The function converts a column with a date from either int64 or object type
    into separate columns with day - month - year
    user can choose to add weekday - hour - minutes columns

    Keyword arguments
        - df (Pandas DataFrame type)-- is the given dataframe

        - column (string)-- the chosen column to create new columns from

        - weekday (boolean)-- True if user wants new column with weekday value
                                by default False

        - hour_minutes (boolean)-- True if user wants two new columns with hour and minutes values
                                    by default False

        - from_type (string)-- 'object' by default if original column type is object and
                                'int64' if original column type is int64

    return: the resulting dataframe with the new colum
    """

    if from_type == 'int64':
        column  = pd.to_datetime(df[column].astype(str))
    else:
        column  = pd.to_datetime(df[column])
        
    datetime = pd.DataFrame(column)

    datetime['day'] = column.dt.day
    datetime['month'] = column.dt.month
    datetime['year'] = column.dt.year

    if weekday == True:
        datetime['weekday'] = column.dt.weekday

    if hour_minutes == True:
        datetime['hour'] = column.dt.hour
        datetime['minutes'] = column.dt.minute

    df = pd.concat([df, datetime], axis = 1)
    df = df.loc[:,~df.columns.duplicated(keep='last')]
    
    return df

def missing_data(df, columns = None, method = 'delete'):
    """
    The function allows user to decide whether they want to delete,
    convert to zeros, or replace by the mean / mode / median all missing 
    values in a specific list of columns of the given df.

    Keyword arguments
        - df (Pandas DataFrame type)-- is the given dataframe

        - columns (list)-- the chosen column/s with missing values
    
        - method (string)--    'delete' drops the missing values (by default)
                                'zero' fills missing values with a zero
                                'mean' fills missing values with the mean
                                'mode' fills missing values with the mode
                                'median' fills missing values with the median

    return-- the resulting dataframe
    """
 
    if method == 'delete':
        df.dropna(subset=columns, inplace= True)    
    elif method == 'zero':
        for column in columns:
            df[column] = df[column].fillna(0)
    elif method == 'mean':
        for column in columns:
            df[column] = df[column].fillna(df[column].mean())
    elif method == 'mode':
        for column in columns:
            df[column] = df[column].fillna(df[column].mode()[0])
    elif method == 'median':
        for column in columns:
            df[column] = df[column].fillna(df[column].median())
    return df