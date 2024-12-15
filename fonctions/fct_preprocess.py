import pandas as pd


def assign_borough(df):
    """
    
    """
    borough_mapping = {
        "Manhattan": ["Manhattan", "New York County"],
        "Brooklyn": ["Brooklyn", "Kings County"],
        "Queens": ["Queens", "Queens County"],
        "Bronx": ["Bronx", "The Bronx", "Bronx County"],
        "Staten Island": ["Staten Island", "Richmond County"]
    }

    def map_borough(row):
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
    df = df.drop(['BROKERTITLE','STATE','LATITUDE', 'LONGITUDE','ADDRESS','LOCALITY', 'SUBLOCALITY', 'STREET_NAME', 'ADMINISTRATIVE_AREA_LEVEL_2', 'LONG_NAME','FORMATTED_ADDRESS','MAIN_ADDRESS'], axis=1, inplace=True)
    print("\n*** Drop Useless Columns OK ***")


def low_modalities_grouping(df):
    """ 
    Compte les occurrences de chaque modalité dans la colonne 'TYPE'
    identifie les modalités avec moins de 50 occurrences
    Remplace les modalités avec moins de 50 occurrences par "Autre_Type"
    """
    type_counts = df['TYPE'].value_counts()
    low_count_types = type_counts[type_counts < 50].index
    df['TYPE'] = df['TYPE'].replace(low_count_types, 'Autre_Type')
    print("\n*** Low Modalities Grouping  OK ***")


def filter_price_range(df, column='PRICE', min_price=49000, max_price=180000000):
    """
    Filtre directement le DataFrame passé en paramètre en supprimant les lignes où les valeurs
    de la colonne spécifiée ne sont pas dans l'intervalle [min_price, max_price].

    :param df: DataFrame à modifier.
    :param column: Nom de la colonne contenant les valeurs à filtrer (par défaut 'PRICE').
    :param min_price: Valeur minimale incluse (par défaut 49000).
    :param max_price: Valeur maximale incluse (par défaut 180000000).
    """
    # Appliquer le filtre directement au DataFrame
    df.drop(df[(df[column] < min_price) | (df[column] > max_price)].index, inplace=True)
    print("\n*** Filter Price Range OK ***")

def round_and_convert_to_int(df, columns):
    """
    Arrondit les valeurs des colonnes spécifiées à l'unité et les convertit en entier.

    :param df: DataFrame à modifier.
    :param columns: Liste des colonnes à traiter.
    """
    for column in columns:
        # Arrondir les valeurs et les convertir en int
        df[column] = df[column].round(0).astype(int)
    
    print("\n*** Round and Convert to Integer  OK ***")
