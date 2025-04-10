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
            return similar.mean()
    return row['passenger_count']