import pandas as pd

def estimate_passenger_count(df, row, vendor, distance_tolerance = 0.5):
    if pd.isna(row['passenger_count']) and row['VendorName'] == vendor:
        # znajdź podobne kursy innych przewoźników
        mask = (
                (df['VendorName'] != vendor) &
                (df['trip_distance'].between(row['trip_distance'] - distance_tolerance,
                                             row['trip_distance'] + distance_tolerance)) &
                (df['passenger_count'].notna())
        )
        similar = df.loc[mask, 'passenger_count']
        if not similar.empty:
            return int(similar.mean())
    return row['passenger_count']

def estimate_distance(df, row, tolerance = 0.5):
     if row['trip_distance'] == 0:
        # znajdź podobne wiersze
        mask = (
                (df['trip_distance'] > 0) &
                (df['tolls_amount'].between(row['tolls_amount'] - tolerance, row['tolls_amount'] + tolerance)) &
                (df['mta_tax'].between(row['mta_tax'] - tolerance, row['mta_tax'] + tolerance)) &
                (df['fare_amount'].between(row['fare_amount'] - tolerance, row['fare_amount'] + tolerance))
        )
        similar_rows = df.loc[mask, 'trip_distance']
        if not similar_rows.empty:
            return similar_rows.mean()
     return row['trip_distance']  # jeśli już jest >0 lub brak podobnych