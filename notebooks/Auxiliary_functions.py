import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from matplotlib.ticker import PercentFormatter
import numpy as np

def inspect_outliers(dataframe,column,whisker_width=1.5):
    '''
    Função utilizada para verificar outliers
    Utilizando fórmula básica que define o boxplot
    
    retorna o dataframe filtrado com os outliers
    '''
    q1= dataframe[column].quantile(0.25)
    q3= dataframe[column].quantile(0.75)
    iqr= q3-q1
    lower_bound = q1-(whisker_width*iqr)
    upper_bound = q3+(whisker_width*iqr)
    
    return dataframe[
          (dataframe[column] < lower_bound) | (dataframe[column] > upper_bound)
    ]

def pairplot(dataframe,columns,hue_column=None,alpha=0.5,corner=True,palette='tab10'):
    '''
    Visualização do Pairplot com hiperparametros citados
    '''
    analysis=columns.copy() + [hue_column]
    sns.pairplot(
    dataframe[analysis],
    diag_kind='kde',
    hue=hue_column,
    plot_kws=dict(alpha=alpha),
    corner=corner,
    palette=palette
);
    
def plot_elbow_silhouette(X,random_state=42,range_k=(2,11)):
    '''
    Visualização de 2 gráficos para Elbow e Silhouette utilizando Kmeans
    '''
    fig, axs = plt.subplots(ncols=2, figsize=(15, 5), tight_layout=True)

    elbow = {}
    silhouette = []
    k_range = range(*range_k)


    for i in k_range:
        kmeans = KMeans(n_clusters=i, random_state=random_state, n_init=10)
        kmeans.fit(X)
        elbow[i] = kmeans.inertia_
        labels = kmeans.labels_
        silhouette.append(silhouette_score(X, labels))

    sns.lineplot(x=list(elbow.keys()), y=list(elbow.values()), ax=axs[0])
    axs[0].set_xlabel('K')
    axs[0].set_xlabel('Inertia')
    axs[0].set_title('Elbow Method')
    sns.lineplot(x=list(k_range), y=silhouette, ax=axs[1])
    axs[1].set_xlabel('K')
    axs[1].set_xlabel('Silhouette Score')
    axs[1].set_title('Silhouette Method')
   
    plt.show()
    
def plot_columns_percent_cluster(dataframe,columns,row_cols=(2,3),figsize=(15,8),column_cluster='Cluster'):
    '''
    Visualização de gráfico de barras classificado por Cluster
    '''
    #Importando a figura
    fig, axs = plt.subplots(nrows=row_cols[0],ncols=row_cols[1],figsize=figsize, sharey=True)
    #Colocando em cada eixo
    for ax, col in zip(axs.flatten(),columns):
        h = sns.histplot(data=dataframe,x=column_cluster,ax=ax,hue=col,multiple='fill',stat='percent',discrete=True,shrink=0.8)
        #Alterando parametros do gráfico
        n_clusters = dataframe[column_cluster].nunique()
        h.set_xticks(range(n_clusters))
        h.yaxis.set_major_formatter(PercentFormatter(1))
        h.set_ylabel('')
        h.tick_params(axis='both', which='both', length=0)
        #Colocando o a % nos graficos
        for bars in h.containers:
            h.bar_label(bars, label_type='center', labels=[f"{b.get_height():.1%}" for b in bars], color='white',fontsize=11)
        #Retirando a linha que divide os gráficos na mesma coluna
        for bar in h.patches:
            bar.set_linewidth(0)

    
        #Espaçando o gráfico
        plt.subplots_adjust(hspace=0.3,wspace=0.3)
    plt.show()
    
def plot_columns_percent_hue_cluster(dataframe,columns,row_cols=(2,3),figsize=(15,8),column_cluster='Cluster',palette='tab10'):
    '''
    Visualização de gráfico de barras com a porcentagem de cada valor com Cluster como hue
    '''
    #Importando a figura
    fig, axs = plt.subplots(nrows=row_cols[0],ncols=row_cols[1],figsize=figsize, sharey=True)
    if not isinstance (axs, np.ndarray):
        axs=np.array(axs)
    #Colocando em cada eixo
    for ax, col in zip(axs.flatten(),columns):
        h = sns.histplot(data=dataframe,x=col,ax=ax,hue=column_cluster,multiple='fill',stat='percent',discrete=True,shrink=0.8,palette=palette)
        #Alterando parametros do gráfico
        n_clusters = dataframe[column_cluster].nunique()
        
        if dataframe[col].dtype!= 'object':
            h.set_xticks(range(dataframe[col].nunique()))
        
        h.yaxis.set_major_formatter(PercentFormatter(1))
        h.set_ylabel('')
        h.tick_params(axis='both', which='both', length=0)
        #Colocando o a % nos graficos
        for bars in h.containers:
            h.bar_label(bars, label_type='center', labels=[f"{b.get_height():.1%}" for b in bars], color='white',fontsize=11)
        #Retirando a linha que divide os gráficos na mesma coluna
        for bar in h.patches:
            bar.set_linewidth(0)
        
        #Remover legenda de cada imagem
        legend = h.get_legend()
        legend.remove()  
    #Espaçando o gráfico
    plt.subplots_adjust(hspace=0.3,wspace=0.3)
    #Colocar legenda geral
    labels=[text.get_text() for text in legend.get_texts()]
    fig.legend(handles=legend.legend_handles,labels=labels,loc='upper center',ncols=dataframe[column_cluster].nunique(),title='Clusters')
    
    plt.show()
def plot_cluster(
    dataframe,
    columns,
    n_colors,
    centroids,
    show_centroide=True, 
    show_points=False,
    column_clusters=None,
):

    #Gráfico em 2d
    #%matplotlib ipympl
    ax = plt.figure().add_subplot()
    #Deixar as cores dos demais pontos iguais as do centroide
    from matplotlib.colors import ListedColormap
    colors= plt.cm.tab10.colors[:n_colors]
    colors=ListedColormap(colors)
    #Pontos do cluster
    x=dataframe[columns[0]]
    y=dataframe[columns[1]]
    #Escolher o que mostrar no grafico
    ligar_centroide = show_centroide
    ligar_pontos= show_points
    #Looping para ingressar os pontos no grafico
    for i, centroid in enumerate(centroids):
        #Decidir mostrar os pontos
        if show_centroide:
            #Pontos Centroide
            ax.scatter(*centroid,s=500,alpha=0.7)
            ax.text(*centroid, f"{i}",horizontalalignment='center', 
                    verticalalignment='center', fontsize=15)
        if show_points:
            s= ax.scatter(x,y,c=column_clusters, cmap=colors)
            ax.legend(*s.legend_elements(),bbox_to_anchor=(1.4,0.7))
    #legendas    
    ax.set_xlabel([columns[0]])
    ax.set_ylabel([columns[1]])
    ax.set_title('Clusters')
    plt.show()
