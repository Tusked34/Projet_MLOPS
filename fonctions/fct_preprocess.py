import pandas as pd


def assign_borough(df):
    """
    Assigne un arrondissement à chaque ligne de données dans le DataFrame.

    Cette fonction prend un DataFrame `df` et assigne un arrondissement à chaque ligne en fonction de certains mots-clés 
    présents dans plusieurs colonnes. Elle utilise la fonction interne `map_borough` pour chaque ligne, qui mappe les valeurs 
    de la ligne (par exemple, "LOCALITY", "SUBLOCALITY", "STREET_NAME", "ADMINISTRATIVE_AREA_LEVEL_2") à un arrondissement 
    défini dans `borough_mapping`. Si aucun mot-clé n'est trouvé, l'arrondissement "autre" est attribué.

    Args:
        df (pandas.DataFrame): Le DataFrame contenant les données à traiter. Une nouvelle colonne 'BOROUGH' sera ajoutée.

    Return:
        None: Le DataFrame est modifié en place et aucune valeur n'est retournée.

    Prints:
        Un message de confirmation indiquant que l'opération a été effectuée avec succès : 
        "*** Grouping Borough OK ***".
    
    """
    borough_mapping = {
        "Manhattan": ["Manhattan", "New York County"],
        "Brooklyn": ["Brooklyn", "Kings County"],
        "Queens": ["Queens", "Queens County"],
        "Bronx": ["Bronx", "The Bronx", "Bronx County"],
        "Staten Island": ["Staten Island", "Richmond County"]
    }

    def map_borough(row):
        """
        Mappe une ligne de données à un arrondissement en fonction de certains mots-clés présents dans plusieurs champs.

        Cette fonction prend un dictionnaire `row` représentant une ligne de données (probablement d'une base de données ou d'un fichier CSV) 
        et tente de faire correspondre des valeurs spécifiques (nom de localité, sous-localité, nom de rue, ou niveau administratif) 
        avec des mots-clés définis pour chaque arrondissement dans `borough_mapping`. Si l'un des mots-clés correspond à un champ de la ligne 
        (parmi "LOCALITY", "SUBLOCALITY", "STREET_NAME", ou "ADMINISTRATIVE_AREA_LEVEL_2"), la fonction retourne le nom de l'arrondissement 
        correspondant. Si aucune correspondance n'est trouvée, la fonction retourne "autre".

        Args:
            row (dict): Un dictionnaire représentant une ligne de données, contenant les clés suivantes : 
                    "LOCALITY", "SUBLOCALITY", "STREET_NAME", et "ADMINISTRATIVE_AREA_LEVEL_2".

        Return:
            str: Le nom de l'arrondissement correspondant si un mot-clé est trouvé, sinon "autre".

        Prints:
            Rien n'est explicitement imprimé par la fonction.
        """
        for borough, keywords in borough_mapping.items():
            if any(x in keywords for x in [
                row.get("LOCALITY"), 
                row.get("SUBLOCALITY"), 
                row.get("STREET_NAME"), 
                row.get("ADMINISTRATIVE_AREA_LEVEL_2")
            ]):
                return borough
        return "autre"

    # Apply the mapping function and update the DataFrame
    df["BOROUGH"] = df.apply(map_borough, axis=1)
    print("\n*** Grouping Borough OK ***")


def drop_useless_columns(df):
    """
    Supprime les colonnes inutiles d'un DataFrame.

    Cette fonction prend un DataFrame `df` et supprime un ensemble de colonnes spécifiques qui sont considérées comme inutiles pour 
    l'analyse ou le traitement ultérieur. Elle modifie le DataFrame en place et n'en retourne pas de copie. 
    Après avoir effectué l'opération, un message de confirmation est imprimé.

    Args:
        df (pandas.DataFrame): Le DataFrame contenant les données à traiter.

    Return:
        None: La fonction modifie le DataFrame en place et ne retourne rien.

    Prints:
        Un message de confirmation indiquant que les colonnes inutiles ont été supprimées avec succès : 
        "*** Drop Useless Columns OK ***".
    """
    df = df.drop(['BROKERTITLE','STATE','LATITUDE', 'LONGITUDE','ADDRESS','LOCALITY', 'SUBLOCALITY', 'STREET_NAME', 'ADMINISTRATIVE_AREA_LEVEL_2', 'LONG_NAME','FORMATTED_ADDRESS','MAIN_ADDRESS'], axis=1, inplace=True)
    print("\n*** Drop Useless Columns OK ***")


def low_modalities_grouping(df):
    """ 
    Regroupe les modalités ayant peu d'occurrences dans la colonne 'TYPE'.

    Cette fonction compte les occurrences de chaque modalité dans la colonne 'TYPE' d'un DataFrame `df`, 
    identifie les modalités ayant moins de 50 occurrences, puis remplace ces modalités par la valeur "Autre_Type".
    Après avoir effectué l'opération, un message de confirmation est imprimé.

    Args:
        df (pandas.DataFrame): Le DataFrame contenant la colonne 'TYPE' à traiter.

    Return:
        None: La fonction modifie le DataFrame en place et ne retourne rien.

    Prints:
        Un message de confirmation indiquant que les modalités avec peu d'occurrences ont été regroupées : 
        "*** Low Modalities Grouping OK ***".
    """
    type_counts = df['TYPE'].value_counts()
    low_count_types = type_counts[type_counts < 50].index
    df['TYPE'] = df['TYPE'].replace(low_count_types, 'Autre_Type')
    print("\n*** Low Modalities Grouping  OK ***")


def filter_price_range(df, column='PRICE', min_price=49000, max_price=180000000):
    """
    Filtre un DataFrame en supprimant les lignes où les valeurs de la colonne spécifiée ne sont pas dans l'intervalle [min_price, max_price].

    Cette fonction filtre les lignes du DataFrame `df` en supprimant celles où les valeurs de la colonne spécifiée 
    (par défaut 'PRICE') sont inférieures à `min_price` ou supérieures à `max_price`. Elle modifie le DataFrame en place 
    et n'en retourne pas de copie. Un message de confirmation est imprimé après l'opération.

    Args:
        df (pandas.DataFrame): Le DataFrame à filtrer.
        column (str, optional): Le nom de la colonne contenant les valeurs à filtrer (par défaut 'PRICE').
        min_price (int, optional): La valeur minimale incluse dans le filtre (par défaut 49000).
        max_price (int, optional): La valeur maximale incluse dans le filtre (par défaut 180000000).

    Return:
        None: La fonction modifie le DataFrame en place et ne retourne rien.

    Prints:
        Un message de confirmation indiquant que le filtrage a été effectué avec succès : 
        "*** Filter Price Range OK ***".
    """
    # Appliquer le filtre directement au DataFrame
    df.drop(df[(df[column] < min_price) | (df[column] > max_price)].index, inplace=True)
    print("\n*** Filter Price Range OK ***")

def round_and_convert_to_int(df, columns):
    """
    Arrondit les valeurs des colonnes spécifiées à l'unité et les convertit en entier.

    Cette fonction arrondit les valeurs dans les colonnes spécifiées de `df` à l'unité la plus proche 
    et les convertit en entier. Le DataFrame est modifié en place et aucune copie n'est retournée. 
    Un message de confirmation est imprimé après l'opération.

    Args:
        df (pandas.DataFrame): Le DataFrame à modifier.
        columns (list of str): Liste des noms des colonnes à traiter.

    Return:
        None: La fonction modifie le DataFrame en place et ne retourne rien.

    Prints:
        Un message de confirmation indiquant que l'opération a été effectuée avec succès : 
        "*** Round and Convert to Integer OK ***".
    """
    for column in columns:
        # Arrondir les valeurs et les convertir en int
        df[column] = df[column].round(0).astype(int)
    
    print("\n*** Round and Convert to Integer  OK ***")
