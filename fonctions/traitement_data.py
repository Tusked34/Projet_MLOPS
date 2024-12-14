import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_missing_and_distinct(df):
    """
    Calcule les valeurs manquantes et distinctes pour chaque colonne du DataFrame.
    Retourne deux DataFrames : un pour les valeurs brutes et un pour les pourcentages.
    """
    missing_values = df.isnull().sum()  # Nombre de valeurs manquantes
    distinct_values = df.nunique()  # Nombre de valeurs distinctes
    total_rows = len(df)

    # Calcul des pourcentages de valeurs manquantes et distinctes (arrondis à l'unité près)
    missing_percentage = (missing_values / total_rows) * 100
    distinct_percentage = (distinct_values / total_rows) * 100

    # Combiner les résultats dans des DataFrames
    summary_df = pd.DataFrame({
        'Missing Values': missing_values,
        'Distinct Values': distinct_values
    })

    percentage_df = pd.DataFrame({
        'Missing Percentage': missing_percentage.round(0),  # Arrondi à l'unité
        'Distinct Percentage': distinct_percentage.round(0)  # Arrondi à l'unité
    })

    return summary_df, percentage_df

def plot_heatmap(data, title, is_percentage=False):
    """
    Génère et affiche une heatmap à partir du DataFrame fourni.
    """
    plt.figure(figsize=(10, 6))
    fmt = ".0f" if is_percentage else "d"
    sns.heatmap(data, annot=True, cmap="coolwarm", fmt=fmt, cbar=False)
    plt.title(title)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.gca().xaxis.set_ticks_position('top')
    plt.gca().yaxis.set_ticks_position('left')
    plt.show()

def analyze_and_plot(df):
    """
    Calcule les valeurs manquantes et distinctes, puis affiche les heatmaps correspondantes.
    """
    # Calculer les DataFrames des valeurs brutes et des pourcentages
    summary_df, percentage_df = calculate_missing_and_distinct(df)
    
    # Afficher les heatmaps
    plot_heatmap(summary_df, 'Missing and Distinct Values per Column')
    plot_heatmap(percentage_df, 'Missing and Distinct Percentage per Column', is_percentage=True)
