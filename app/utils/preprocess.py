from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_colwidth', None)

def drop_and_rename(dat, col_name1, col_name2, col_change_dict):
    """
    Drop two certain columns change name of last one to constant name 
    """
    
    dat  = dat.drop([col_name1, col_name2], axis = 1)
    
    return dat.rename(columns = col_change_dict)

def concat_dfs(dat,outline_col_lst, change_dict):
    """
    Concatenate all outline columns into one 
    """
    data_lst = [
        drop_and_rename(dat,outline_col_lst[0],outline_col_lst[1], change_dict),
        drop_and_rename(dat,outline_col_lst[0],outline_col_lst[2], change_dict),
        drop_and_rename(dat,outline_col_lst[1],outline_col_lst[2], change_dict)
                ]
    return pd.concat(data_lst)

def split_data_add_type(dat, col_type_str):
    
    """
    Split data and create new column which set constant split
    """
    
    train, dev_test = train_test_split(dat, test_size = 0.3, random_state = 42)
    dev, test = train_test_split(dev_test, test_size = 0.5, random_state = 42)
    
    train[col_type_str] = "train"
    dev[col_type_str] = "dev"
    test[col_type_str] = "test"
    
    return pd.concat([train, dev, test])   

def set_column_text_len(dat, col_name):
    """
    Get number of words in selected columns for each record
    """
    dat['{}TokensLength'.format(col_name)] = dat[col_name].str.split(" ").str.len()
    return dat 
def plot_column_distribution(dat, col_name):
    """
    plot distribution of given columns 
    """
    return sns.distplot(dat[col_name], hist=True, kde=False, 
             bins=int(len(dat)/10), color = 'lightblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 5}) 