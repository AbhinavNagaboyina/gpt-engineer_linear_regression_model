import pandas as pd
from sklearn.preprocessing import StandardScaler

def S_scaler(df):
    advertising_data = df
# Initialize the standard scaler
    scaler = StandardScaler()

    # Apply standard scaling to the dataset
    advertising_scaled = scaler.fit_transform(advertising_data)

    # Convert the scaled data back to a DataFrame for better visualization
    advertising_scaled_df = pd.DataFrame(advertising_scaled, columns=advertising_data.columns)


    scaler_df = advertising_scaled_df  # For simplicity, we don't do any feature engineering

    return scaler_df
