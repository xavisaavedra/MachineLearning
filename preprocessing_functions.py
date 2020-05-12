# Version library:  0.0.1
# =======================

# import pathlib

import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer

# import math

import matplotlib.pyplot as plt
import seaborn as sns

import config
import chardet
# import glob

# import joblib


# Individual pre-processing and training functions
# ================================================


# check the file encoding and load it into a pandas dataframe
# ===========================================================

def load_dataset(*, file_name: str) -> pd.DataFrame:
    # Load a persisted pipeline
    print("File".ljust(45), "Encoding")
    filename = f"{config.DATASET_DIR}/{file_name}"

    with open(filename, 'rb') as rawdata:
        result = chardet.detect(rawdata.read())
        print(file_name.ljust(45), result['encoding'])
        print('Cargando dataset...')

    _data = pd.read_csv(filename, encoding=result['encoding'])
    return _data


def get_numerical_vars(df):

    # make list of numerical variables
    num_vars = [var for var in df.columns if df[var].dtypes != 'O']
    # print('Número de variables numéricas: ', len(num_vars))

    # visualise the numerical variables
    return df[num_vars]


def get_discrete_vars(df):

    num_vars = get_numerical_vars(df)
    #  let's male a list of discrete variables
    discrete_vars = [var for var in num_vars if len(
        df[var].unique()) < 20]

    # print('Número de variables discretas: ', len(discrete_vars))
    return df[discrete_vars]


def get_continous_vars(df):
    # make list of continuous variables
    num_vars = get_numerical_vars(df)
    discrete_vars = get_discrete_vars(df)
    cont_vars = [
        var for var in num_vars if var not in discrete_vars]

    # print('Number of continuous variables: ', len(cont_vars))
    return df[cont_vars]


def get_cat_vars(df):
    # capture categorical variables in a list
    cat_vars = [var for var in df.columns if df[var].dtypes == 'O']

    # print('Number of categorical variables: ', len(cat_vars))
    return df[cat_vars]


def redefine_columns(df):
    # Map the lowering function to all column names
    df.columns = map(str.lower, df.columns)
    df.columns = df.columns.str.replace(' ', '')
    print(f'Columnas: {list(df.columns)}')
    return df


def interpreting_skew(df):
    # skewness
    skew = df.skew()

    # Interpreting Skew
    if -0.5 < skew < 0.5:
        print(f'Un sesgo de {skew} distribución es aprox. simétrica')
    elif -0.5 < skew < -1.0 or 0.5 < skew < 1.0:
        print(f'Un sesgo de {skew} distribución es moderadamente sesgada')
    else:
        print(f'Un sesgo de {skew} distribución está muy sesgada')


def interpreting_kurtosis(df):
    kurtosis = df.kurtosis()

# Interpreting Kurtosis
    if -0.5 < kurtosis < 0.5:
        print(f'Una curtosis de {kurtosis} distribución es aprox. normal')
    elif kurtosis <= -0.5:
        print(f'Una curtosis de {kurtosis} dist. es de cola ligera (neg.)')
    elif kurtosis >= 0.5:
        print(f'Una curtosis de {kurtosis} dist. es de cola gruesa (pos.)')


def analisis_num_val(df):
    plt.figure(figsize=(20, 6))

    """ Creamos el primer grafico con este formato
    plt.subplot(filas, columnas, No de grafico) """

    plt.subplot(131)
    # Density Plot and Histogram of all arrival delays
    sns.distplot(df, hist=True, kde=True,
                 bins='auto',
                 hist_kws={'edgecolor': 'black'},
                 kde_kws={'linewidth': 4})

    plt.subplot(132)
    sns.distplot(df, hist=False, kde=True,
                 kde_kws={'shade': True, 'linewidth': 3})

    plt.subplot(133)
    sns.boxplot(y=df)


# Function to calculate missing values by column
def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    # Print some summary information

    line1 = "Your dataframe has " + str(df.shape[1]) + " columns."
    line2 = "There are " + str(mis_val_table_ren_columns.shape[0])
    line3 = " columns that have missing values."

    print(line1)
    print(line2 + line3)

    # Return the dataframe with missing information
    return mis_val_table_ren_columns


""" Detectamos valores 'Outliers'. Esta funcion
    nos devuelve los indices en el dataframe
    De los esos valores """


def drop_outliers(df):
    quartile_1, quartile_3 = np.percentile(df, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((df > upper_bound) | (df < lower_bound))


def apply_yeojohnson(df):
    feature = pd.DataFrame(df)
    name = feature.columns
    print(name)
    pt = PowerTransformer(method='yeo-johnson', standardize=True,)
    tr_yeo = pt.fit_transform(feature)
    return pd.DataFrame(tr_yeo, columns=name)
