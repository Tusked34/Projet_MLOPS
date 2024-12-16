import pandas as pd
import math
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
    non_numeric_columns = df.select_dtypes(exclude=['number']).columns
    
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

def plot_density(df):
    """
    
    """
    # Liste des colonnes numériques
    numeric_columns = df.select_dtypes(include=['number']).columns
    # Configuration de l'affichage
    sns.set(style="whitegrid")
    # Calcul du nombre de lignes et de colonnes pour le sous-graphique
    num_cols = 4  # Par exemple, on fixe 4 colonnes
    num_rows = math.ceil(len(numeric_columns) / num_cols)

    plt.figure(figsize=(15, 5 * num_rows))

    # Création de graphiques pour chaque colonne numérique avec courbe de densité
    for i, column in enumerate(numeric_columns, 1):
        plt.subplot(num_rows, num_cols, i)
        sns.kdeplot(df[column], fill=True, color='blue')
        plt.title(f'Density Plot of {column}')
        plt.xlabel(column)
        plt.ylabel('Density')

    plt.tight_layout()
    plt.show()

    # Boîtes à moustaches pour les mêmes colonnes
    plt.figure(figsize=(15, 5 * num_rows))
    for i, column in enumerate(numeric_columns, 1):
        plt.subplot(num_rows, num_cols, i)
        sns.boxplot(x=df[column], color='orange')
        plt.title(f'Boxplot of {column}')
        plt.xlabel(column)

    plt.tight_layout()
    plt.show()