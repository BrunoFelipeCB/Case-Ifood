import seaborn as sns

def inspect_outliers(dataframe,column,whisker_width=1.5):
    q1= dataframe[column].quantile(0.25)
    q3= dataframe[column].quantile(0.75)
    iqr= q3-q1
    lower_bound = q1-(whisker_width*iqr)
    upper_bound = q3+(whisker_width*iqr)
    
    return dataframe[
          (dataframe[column] < lower_bound) | (dataframe[column] > upper_bound)
    ]

def pairplot(dataframe,columns,hue_column=None,alpha=0.5,corner=True):
    analysis=columns.copy() + [hue_column]
    sns.pairplot(
    dataframe[analysis],
    diag_kind='kde',
    hue=hue_column,
    plot_kws=dict(alpha=alpha),
    corner=corner
);