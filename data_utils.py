import pandas as pd

def estimate_passenger_count(df: pd.DataFrame, row: pd.Series, vendor: str, distance_tolerance: float = 0.5) -> float:
    """
    Estymuje brakującą wartość 'passenger_count' dla wskazanego dostawcy systemu TPEP,
    bazując na podobnych kursach z innych firm.

    Parametry:
    - df: Główna ramka danych zawierająca wszystkie kursy.
    - row: Wiersz, dla którego próbujemy oszacować liczbę pasażerów.
    - vendor: Nazwa dostawcy systemu TPEP, którego braki w liczbie pasażerów próbujemy uzupełnić.
    - distance_tolerance: Maksymalne odchylenie dla podobnej długości trasy (domyślnie 0,5 mili).

    Zwraca:
    - Oszacowana lub pierwotna wartość 'passenger_count'.
    """
    if pd.isna(row['passenger_count']) and row['VendorName'] == vendor:
        # Szukanie kursów innych dostawców systemów TPEP o podobnej długości trasy
        mask = (
                (df['VendorName'] != vendor) &
                (df['trip_distance'].between(row['trip_distance'] - distance_tolerance,
                                             row['trip_distance'] + distance_tolerance)) &
                (df['passenger_count'].notna())
        )
        similar = df.loc[mask, 'passenger_count']
        if not similar.empty:
            return int(similar.mean()) # Zwraca uśrednioną liczbę pasażerów (zrzutowaną na liczbę całkowitą)
    return row['passenger_count'] # Jeśli brak podobnych wierszy – zwraca oryginalną wartość

def estimate_distance(df: pd.DataFrame, row: pd.Series, tolerance: float = 0.5) -> float:
    """
    Estymuje brakującą wartość 'trip_distance' dla kursu z wartością 0,
    bazując na podobnych kursach zbliżonych pod względem ceny, podatków i opłat drogowych.

    Parametry:
    - df: Główna ramka danych zawierająca wszystkie kursy.
    - row: Wiersz, dla którego próbujemy oszacować długość trasy.
    - tolerance: Maksymalne odchylenie dla podobieństwa do innych wierszy (domyślnie 0.5 jednostki w zależności od kolumny).

    Zwraca:
    - Oszacowana lub pierwotna wartość 'trip_distance'.
    """

    if row['trip_distance'] == 0:
        # Szukanie podobnych kursów z niezerowym dystansem i zbliżonymi opłatami
        mask = (
                (df['trip_distance'] > 0) &
                (df['tolls_amount'].between(row['tolls_amount'] - tolerance, row['tolls_amount'] + tolerance)) &
                (df['mta_tax'].between(row['mta_tax'] - tolerance, row['mta_tax'] + tolerance)) &
                (df['fare_amount'].between(row['fare_amount'] - tolerance, row['fare_amount'] + tolerance))
        )
        similar_rows = df.loc[mask, 'trip_distance']
        if not similar_rows.empty:
            return similar_rows.mean() # Zwraca uśrednioną długość podobnych kursów
    return row['trip_distance'] # Jeśli brak podobnych wierszy lub długość kursu jest niezerowa – zwraca oryginalną wartość