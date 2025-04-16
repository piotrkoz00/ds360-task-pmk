import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.metrics import PredictionErrorDisplay
from scipy.stats import probplot

def histogram(df: pd.DataFrame, col: str, title: str, bins: int = 50, log_scale: bool = False):
    """
    Wyświetla histogram danej kolumny.

    Parametry:
    - df: Ramka danych (DataFrame) z danymi wejściowymi.
    - col: Nazwa kolumny, której histogram ma zostać narysowany.
    - title: Tytuł wykresu.
    - bins: Liczba przedziałów w histogramie (domyślnie 50).
    - log_scale: Flaga wskazująca na wykorzystanie skali logarytmicznej na osi OX (domyślnie False).
    """
    sns.histplot(data = df[col], bins=bins, log_scale=log_scale)
    plt.title(title)
    plt.show()

def boxplot(df: pd.DataFrame, x: str, y: str, title: str, color: str = 'blue', log_scale: bool = False):
    """
    Wyświetla wykres pudełkowy (ramka-wąsy) wartości y względem x.

    Parametry:
    - df: Ramka danych (DataFrame) z danymi wejściowymi.
    - x: Zmienna kategoryczna (np. grupująca).
    - y: Zmienna ciągła (wartość mierzona).
    - title: Tytuł wykresu.
    - color: Kolor ramek (domyślnie niebieski).
    - log_scale: Flaga wskazująca na wykorzystanie skali logarytmicznej na osi OX (domyślnie False).
    """
    sns.boxplot(data = df, x=x, y=y, color=color, log_scale=log_scale)
    plt.title(title)
    plt.show()

def barplot(df: pd.DataFrame, x: str, y: str, title: str, log_y: bool = False):
    """
    Wyświetla wykres słupkowy (kolumnowy) dla zmiennych x i y.

    Parametry:
    - df: DataFrame z danymi wejściowymi.
    - x: Zmienna kategoryczna (np. grupująca).
    - y: Zmienna numeryczna.
    - title: Tytuł wykresu.
    - log_y: Flaga wskazująca na wykorzystanie skali logarytmicznej na osi OY (domyślnie False).
    """
    b = sns.barplot(data = df, x=x, y=y)
    if log_y:
        b.set_yscale("log")
    plt.title(title)
    plt.show()

def corr_heatmap(corr_matrix: pd.DataFrame, cmap: str ='inferno', annot: bool = True):
    """
    Rysuje heatmapę macierzy korelacji.

    Parametry:
    - corr_matrix: Macierz korelacji (np. df.corr()).
    - cmap: Styl kolorów (domyślnie 'inferno').
    - annot: Czy wyświetlać liczby na mapie (domyślnie True).
    """
    sns.heatmap(corr_matrix, cmap=cmap, annot=annot, fmt=".2f")
    plt.show()

def lineplot(data, x: str, y: str, title: str, xlabel: str, ylabel: str, marker: str = 'o'):
    """
    Tworzy wykres liniowy z oznaczonymi punktami.

    Parametry:
    - data: Dane pogrupowane w oparciu o ramkę danych i funkcję agregującą.
    - x: Zmienna na osi X.
    - y: Zmienna na osi Y.
    - title: Tytuł wykresu.
    - xlabel: Etykieta osi X.
    - ylabel: Etykieta osi Y.
    - marker: Styl znacznika punktów (domyślnie 'o').
    """
    sns.lineplot(data=data, x=x, y=y, marker=marker)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def model_plots(y_true: np.ndarray | list | pd.Series, y_pred: np.ndarray | list | pd.Series, suptitle: str):
    """
    Wyświetla dwa wykresy oceny modelu regresyjnego:
    - Wartości obserwowane względem wartości przewidywanych
    - Wartości resztowe względem wartości przewidywanych

    Parametry:
    - y_true: Rzeczywiste wartości zmiennej wyjściowej.
    - y_pred: Przewidywane przez model wartości zmiennej wyjściowej.
    - suptitle: Tytuł całego wykresu.
    """
    fig, axes = plt.subplots(ncols=2, figsize=(12, 6))
    PredictionErrorDisplay.from_predictions(y_true=y_true, y_pred=y_pred, kind='actual_vs_predicted', ax=axes[0])
    axes[0].set_title("Wartości obserwowane względem przewidywanych")
    axes[0].set_xlabel("Wartości przewidywane")
    axes[0].set_ylabel("Wartości obserwowane")
    PredictionErrorDisplay.from_predictions(y_true=y_true, y_pred=y_pred, ax=axes[1])
    axes[1].set_title("Wartości resztowe względem przewidywanych")
    axes[1].set_xlabel("Wartości przewidywane")
    axes[1].set_ylabel("Wartości resztowe (obserwowane - przewidywane)")
    fig.suptitle(suptitle)
    plt.tight_layout()
    plt.show()

def qq_plot(residuals: np.ndarray | list | pd.Series):
    """
    Rysuje wykres Q-Q do oceny normalności rozkładu reszt.

    Parametry:
    - residuals: Seria z wartościami reszt modelu (y_true - y_pred).
    """
    plt.figure(figsize=(6, 6))
    probplot(residuals, dist="norm", plot=plt)
    plt.title("Wykres normalności reszt")
    plt.xlabel("Reszty")
    plt.ylabel("Wartość normalna")
    plt.tight_layout()
    plt.show()

