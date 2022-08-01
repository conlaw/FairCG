import numpy as np
import pandas as pd

'''
Helper functions to binerize data columns
'''

def binNumeric(data, column, quantiles = 10, drop_first = False):
    '''
    Bins numeric columns by quantile (optional parameter)
    Returns: Pandas dataframe with one column per bin
    '''
    boolNum = pd.get_dummies(pd.qcut(data[column].values, quantiles, duplicates = 'drop'), 
                          prefix = column, 
                          drop_first = drop_first).astype(np.bool_)
    return pd.concat([boolNum, ~boolNum], axis = 1)
    
def binCategorical(data, column, drop_first = False):
    '''
    Creates one hot encoding for categorical variables
    Returns: Pandas dataframe with one column per bin
    '''
    boolCat = pd.get_dummies(data[column], prefix = column, drop_first = drop_first).astype(np.bool_)
    return pd.concat([boolCat, ~boolCat], axis = 1)
    
def threshNumeric(data, column, quant = 10):
    '''
    Creates binarized numeric features using sequence of thresholds specified by the sample quantiles
    Returns: Pandas dataframe with one column representing > quant and one <= quant for each sample quantile
    '''
    quantiles = np.unique(np.quantile(data[column][data[column].notnull()], np.arange(1/quant,1,1/quant)))
    lowThresh = np.transpose([np.where(data[column] <= x, 1, 0) for x in quantiles])
    highThresh = 1-lowThresh
    
    binarized = pd.DataFrame(np.concatenate((lowThresh, highThresh), axis = 1))
    binarized.columns = [column+'_'+str(round(x,2))+'*' for x in quantiles] + [column+'_'+str(round(x,2))+'_' for x in quantiles]

    return binarized.astype(np.bool_)

def binerizeData(data, binNumeric = False, quantiles = 10, verbose = False):
    '''
    Controller to binerize a whole dataframe.
    Takes arguments:
        data: Dataframe to binerize
        binNumeric: Specifies whether to use binning or thresholding for numeric data
        quantiles: Specifies number of quantiles to use for numeric columns
        verbose: Specifies whether or not you want intermediary updates on conversions
    Returns: Pandas dataframe with one column representing > quant and one <= quant for each sample quantile
    '''

    #Set function to deal with numeric columns
    numFun = binNumeric if binNumeric else threshNumeric

    binerizedDF = pd.DataFrame()

    #Loop through and binerize columns
    for col in data:
        if data[col].dtype == 'object':
            if verbose: print('Column '+str(col)+': Applying 1 Hot Encoding.')
            new_cols = binCategorical(data, col)
        elif data[col].dtype in ['float64','int64']:
            if verbose: print('Column '+str(col)+': Binarzing with specified numeric strategy.')
            new_cols = numFun(data, col)
        elif data[col].dtype == 'bool':
            if verbose: print('Column '+str(col)+': Copying to final dataframe.')
            new_cols = data[col]
        else:
            warnings.warn('Column '+str(col)+' has unexpected data type '+str(data[col].dtype)+' not including in final dataframe.')
    
        binerizedDF = pd.concat([binerizedDF, new_cols], axis = 1)
    
    return binerizedDF



                            