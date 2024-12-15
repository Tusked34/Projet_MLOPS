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
    return df


def drop_useless_columns(df):
    df = df.drop(['BROKERTITLE','STATE','LATITUDE', 'LONGITUDE','ADDRESS','LOCALITY', 'SUBLOCALITY', 'STREET_NAME', 'ADMINISTRATIVE_AREA_LEVEL_2', 'LONG_NAME','FORMATTED_ADDRESS','MAIN_ADDRESS'], axis=1, inplace=True)


def low_modalities_grouping(df):
    """ 
    Compte les occurrences de chaque modalité dans la colonne 'TYPE'
    identifie les modalités avec moins de 50 occurrences
    Remplace les modalités avec moins de 50 occurrences par "Autre_Type"
    """
    type_counts = df['TYPE'].value_counts()
    low_count_types = type_counts[type_counts < 50].index
    df['TYPE'] = df['TYPE'].replace(low_count_types, 'Autre_Type')