import re
import numpy as np
import pandas as pd

from pandas.api.types import is_string_dtype, is_numeric_dtype

def display_all(dataframe):
    """
    Displays maximum of 1000 rows and columns for a given data frame.

    Example:
    --------
    >>> df = pd.DataFrame({'col1': ['A', 'B', 'C'], 'col2': [1, 2, 3]})
    >>> display_all(df)
    	col1 col2
    0	A	 1
    1	B	 2
    2	C	 3
    """
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000):
        display(dataframe)

        
def extract_date_features(dataframe, field_names, drop=True, time=False, errors="raise"):
    """
    extract_date_features converts date (dtype = datetime64) column to many columns containing information about the date. Changes are applied inplace.
        
    Parameters:
    -----------
    dataframe: A pandas dataframe.
    field_names: A string or list of strings that is the name of the date column you wish to expand/extract. If it is not a datetime64 series, 
    it will be converted to one with pd.to_datetime.
    drop: If true then the original date column will be removed.
    time: If true time features: Hour, Minute, Second will be added.
    
    Examples:
    ---------
    >>> df = pd.DataFrame({ 'A' : pd.to_datetime(['3/11/2000', '3/12/2000', '3/13/2000'], infer_datetime_format=False) })
    >>> df
        A
    0   2000-03-11
    1   2000-03-12
    2   2000-03-13
    >>> extract_date_features(df, 'A')
    >>> df
        A_year A_month A_week A_day A_dayofweek A_dayofyear A_is_month_end A_is_month_start A_is_quarter_end A_is_quarter_start A_is_year_end A_is_year_start
    0   2000   3       10     11    5           71          False          False            False            False              False         False          
    1   2000   3       10     12    6           72          False          False            False            False              False         False          
    2   2000   3       11     13    0           73          False          False            False            False              False         False          
    """
    if isinstance(field_names, str):
        field_names = [field_names]

    for field_name in field_names:
        field = dataframe[field_name]
        field_type = field.dtype

        if isinstance(field_type, pd.core.dtypes.dtypes.DatetimeTZDtype):
            field_type = np.datetime64

        if not np.issubdtype(field_type, np.datetime64):
            dataframe[field_name] = field = pd.to_datetime(field, infer_datetime_format=True, errors=errors)
        
        prefix = re.sub("[Dd]ate$", '', field_name)

        if not prefix.endswith("_"):
            prefix += "_"

        attributes = ['year', 'month', 'week', 'day', 'dayofweek', 'dayofyear', 'is_month_end', 'is_month_start', 'is_quarter_end', 
        'is_quarter_start', 'is_year_end', 'is_year_start']

        if time:
            attributes += ['hour', 'minute', 'second']
        
        for attr in attributes:
            dataframe[prefix + attr] = getattr(field.dt, attr)

        if drop:
            dataframe.drop(field_name, axis=1, inplace=True)


def convert_to_categorical(dataframe):
    """
    Changes all columns of strings in panda's dataframe to a column of categorical values. Changes are applied inplace.

    Parameters:
    -----------
    dataframe: A pandas dataframe. Any columns of strings will be changed to categorical values.

    Examples:
    ---------
    >>> df = pd.DataFrame({'col1' : [1, 2, 3], 'col2' : ['a', 'b', 'a']})
    >>> df
       col1 col2
    0     1    a
    1     2    b
    2     3    a
    note the type of col2 is string

    >>> convert_to_categorical(df)
    >>> df
       col1 col2
    0     1    a
    1     2    b
    2     3    a
    now the type of col2 is category
    """
    for feature_name, column_data in dataframe.items():
        if is_string_dtype(column_data):
            dataframe[feature_name] = column_data.astype('category').cat.as_ordered()


def preprocess_dataframe(dataframe, target_field=None, skip_fields=None, ignored_fields=None, na_dict=None, preprocess_function=None, 
        max_n_cat=None, subset=None):
    """
    preprocess_dataframe takes a dataframe and splits off the dependent variable, and changes the dataframe into an entirely numeric dataframe. 
    For each column of the dataframe which is not in skip_fields nor in ignore_fields, NA values are replaced by the median value of the column.

    Parameters:
    -----------
    dataframe: A pandas dataframe. 
    target_field: The name of the dependent variable.
    skip_fields: A list of fields that should be dropped from the dataframe.
    ignored_fields: A list of fields that should be ignored during processing. 
    na_dict: a dictionary with key = feature_name, value = some_value that should replace missing values in the column with name feature_name.
    New column feature_name_na will be added to the dataframe with values True where value was missing.
    preprocess_function: A function that gets applied to the dataframe.
    max_n_cat: The maximum number of categories to break into dummy values, instead of integer codes.
    subset: Takes a random subset of size subset from the dataframe.
    """
    if not ignored_fields:
        ignored_fields = []
    
    if not skip_fields:
        skip_fields = []

    if subset:
        dataframe = get_sample(dataframe, subset)

    ignored_fields = dataframe.loc[:, ignored_fields]
    dataframe.drop(ignored_fields, axis=1, inplace=True)

    if not target_field is None:
        if not is_numeric_dtype(dataframe[target_field]):
            dataframe[target_field] = pd.Categorical(dataframe[target_field]).codes
        target = dataframe[target_field]
        skip_fields += [target_field]

    dataframe.drop(skip_fields, axis=1, inplace=True)

    if na_dict is None: 
        na_dict = {}

    na_dict_initial = na_dict.copy()

    for feature_name, column_data in dataframe.items():
        na_dict = fix_missing(dataframe, column_data, feature_name, na_dict)

    if len(na_dict_initial.keys()) > 0:
        dataframe.drop([name + '_na' for name in list(set(na_dict.keys()) - set(na_dict_initial.keys()))], axis=1, inplace=True)

    for feature_name, column_data in dataframe.items():
        numericalize(dataframe, column_data, feature_name, max_n_cat)

    dataframe = pd.get_dummies(dataframe, dummy_na=True)
    dataframe = pd.concat([ignored_fields, dataframe], axis=1)

    res = [dataframe, target, na_dict]
    return res
    

def get_sample(dataframe, sample_size):
    """ 
    Gets a random sample of sample_size rows from the dataframe, without replacement.
    
    Parameters:
    -----------
    dataframe: A pandas dataframe.
    sample_size: The number of rows you wish to sample.

    Returns:
    --------
    return value: A random sample of sample_size rows of dataframe.

    Examples:
    ---------
    >>> df = pd.DataFrame({'col1' : [1, 2, 3], 'col2' : ['a', 'b', 'a']})
    >>> df
       col1 col2
    0     1    a
    1     2    b
    2     3    a
    >>> get_sample(df, 2)
       col1 col2
    1     2    b
    2     3    a
    """
    idxs = sorted(np.random.permutation(len(dataframe))[:sample_size])
    return dataframe.iloc[idxs].copy()


def fix_missing(dataframe, column_data, feature_name, na_dict):
    """ 
    Fill missing data in a column of dataframe with the median, and add a {feature_name}_na column
    which specifies if the data was missing.

    Parameters:
    -----------
    dataframe: A pandas dataframe.
    column_data: The column of data to fix by filling in missing data.
    feature_name: The name of the new filled column in dataframe.
    na_dict: A dictionary of values to create na's of and the value to insert. If feature_name is not a key of na_dict the median 
    will fill any missing data. Also if feature_name is not a key of na_dict and there is no missing data in column_data, then no 
    {feature_name}_na column is not created.

    Examples:
    ---------
    >>> df = pd.DataFrame({'col1' : [1, np.NaN, 3], 'col2' : [5, 2, 2]})
    >>> df
       col1 col2
    0     1    5
    1   nan    2
    2     3    2
    >>> fix_missing(df, df['col1'], 'col1', {})
    >>> df
       col1 col2 col1_na
    0     1    5   False
    1     2    2    True
    2     3    2   False
    >>> df = pd.DataFrame({'col1' : [1, np.NaN, 3], 'col2' : [5, 2, 2]})
    >>> df
       col1 col2
    0     1    5
    1   nan    2
    2     3    2
    >>> fix_missing(df, df['col2'], 'col2', {})
    >>> df
       col1 col2
    0     1    5
    1   nan    2
    2     3    2
    >>> df = pd.DataFrame({'col1' : [1, np.NaN, 3], 'col2' : [5, 2, 2]})
    >>> df
       col1 col2
    0     1    5
    1   nan    2
    2     3    2
    >>> fix_missing(df, df['col1'], 'col1', {'col1' : 500})
    >>> df
       col1 col2 col1_na
    0     1    5   False
    1   500    2    True
    2     3    2   False
    """
    if is_numeric_dtype(column_data):
        if pd.isnull(column_data).sum() or (feature_name in na_dict):
            dataframe[feature_name + '_na'] = pd.isnull(column_data)
            filler = na_dict[feature_name] if feature_name in na_dict else column_data.median()
            dataframe[feature_name] = column_data.fillna(filler)
            na_dict[feature_name] = filler
    return na_dict


def numericalize(dataframe, column_data, feature_name, max_n_cat):
    """ 
    Changes the column column_data from a categorical type to it's integer codes.
    
    Parameters:
    -----------
    dataframe: A pandas dataframe.
    column_data: The column you wish to change into the categories.
    name: The column name you wish to insert into df. This column will hold the integer codes.
    max_n_cat: If column_data has more categories than max_n_cat it will not change it to its integer codes. 
    If max_n_cat is None, then column_data will always be converted.

    Examples:
    ---------
    >>> df = pd.DataFrame({'col1' : [1, 2, 3], 'col2' : ['a', 'b', 'a']})
    >>> df
       col1 col2
    0     1    a
    1     2    b
    2     3    a
    note the type of col2 is string
    >>> covert_to_categorical(df)
    >>> df
       col1 col2
    0     1    a
    1     2    b
    2     3    a
    now the type of col2 is category {a : 1, b : 2}
    >>> numericalize(df, df['col2'], 'col3', None)
       col1 col2 col3
    0     1    a    1
    1     2    b    2
    2     3    a    1
    """
    if not is_numeric_dtype(column_data) and ( max_n_cat is None or len(column_data.cat.categories) > max_n_cat):
        dataframe[feature_name] = pd.Categorical(column_data).codes + 1
    
    
    

        

        