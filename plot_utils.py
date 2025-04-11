import matplotlib.pyplot as plt
import seaborn as sns

def histogram(df, col, title, bins=50, log_scale=False):
    sns.histplot(data = df[col], bins=bins, log_scale=log_scale)
    plt.title(title)
    plt.show()

def boxplot(df, x, y, title, color='blue', log_scale=False):
    sns.boxplot(data = df, x=x, y=y, color=color, log_scale=log_scale)
    plt.title(title)
    plt.show()

def barplot(df, x, y, title, log_y = False):
    b = sns.barplot(data = df, x=x, y=y)
    if log_y:
        b.set_yscale("log")
    plt.title(title)
    plt.show()

def corr_heatmap(corr_matrix, cmap='inferno', annot=True):
    sns.heatmap(corr_matrix, cmap=cmap, annot=annot, fmt=".2f")
    plt.show()


