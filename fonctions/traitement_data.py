import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_missing_and_distinct(df):
    """
    Calcule les valeurs manquantes et distinctes pour chaque colonne du DataFrame.
    Retourne un DataFrame avec les valeurs brutes.
    """
    missing_values = df.isnull().sum()  # Nombre de valeurs manquantes
    distinct_values = df.nunique()  # Nombre de valeurs distinctes

    # Combiner les résultats dans un DataFrame
    summary_df = pd.DataFrame({
        'Missing Values': missing_values,
        'Distinct Values': distinct_values
    })

    return summary_df

def plot_heatmap(data, title):
    """
    Génère et affiche une heatmap à partir du DataFrame fourni.
    """
    plt.figure(figsize=(10, 6))
    sns.heatmap(data, annot=True, cmap="coolwarm", fmt="d", cbar=False)
    plt.title(title)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.gca().xaxis.set_ticks_position('top')
    plt.gca().yaxis.set_ticks_position('left')
    plt.show()

def analyze_and_plot(df):
    """
    Calcule les valeurs manquantes et distinctes, puis affiche la heatmap correspondante.
    """
    # Calculer le DataFrame des valeurs brutes
    summary_df = calculate_missing_and_distinct(df)
    
    # Afficher la heatmap des valeurs brutes
    plot_heatmap(summary_df, 'Missing and Distinct Values per Column')