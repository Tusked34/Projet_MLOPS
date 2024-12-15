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


def plot_pie_charts(df):
    """
    
    """
    # Sélectionner les colonnes non numériques
    non_numeric_columns = df.select_dtypes(exclude=['float64', 'int64']).columns
    
    # Définir le nombre de colonnes et de lignes en fonction du nombre de colonnes non numériques
    n = len(non_numeric_columns)
    n_rows = (n // 3) + (1 if n % 3 else 0)  # Nombre de lignes, 3 par ligne
    
    # Créer un grid de subplots
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows)) 
    
    # Aplatir l'array des axes pour itérer facilement
    axes = axes.flatten()

    # Loop à travers les colonnes non numériques et afficher les pie charts
    for i, column in enumerate(non_numeric_columns):
        plt.style.use("bmh")
        prop_mod_cah = df[column].value_counts()  # Calcul les proportions d'appartitions des modalités dans la variable
        
        # Afficher chaque graphique dans un subplot
        axes[i].pie(prop_mod_cah, labels=prop_mod_cah.index, autopct='%1.1f%%', startangle=140)
        axes[i].set_title(f"Distribution de {column}")

    # Supprimer les axes non utilisés (si n n'est pas un multiple de 3)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()  # Ajuster l'espacement entre les graphiques
    plt.show()
