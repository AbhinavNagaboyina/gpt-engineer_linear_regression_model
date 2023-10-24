import pandas as pd
import streamlit as st
def clean_data(df):
    advertising_data = df
    missing_values = advertising_data.isnull().sum()
    st.header("Missing_values")
    st.write(missing_values)

    advertising_data.fillna(df.median(), inplace=True)
    st.write("Filled the missing values with median")
    # Remove outliers from continuous variables
    # Calculate IQR for all columns
    
    Q1_advertising = advertising_data.quantile(0.25)
    Q3_advertising = advertising_data.quantile(0.75)
    IQR_advertising = Q3_advertising - Q1_advertising

    # Define bounds for outliers
    lower_bound_advertising = Q1_advertising - 1.5 * IQR_advertising
    upper_bound_advertising = Q3_advertising + 1.5 * IQR_advertising

    # Identify outliers for each column
    outliers_advertising = {}
    for column in advertising_data.columns:
        outliers = advertising_data[(advertising_data[column] < lower_bound_advertising[column]) | 
                                    (advertising_data[column] > upper_bound_advertising[column])]
        outliers_advertising[column] = len(outliers)
    st.header("Outliers")
    st.write(outliers_advertising)

    # Replace outliers in the 'newspaper' column with its median
    median_newspaper = advertising_data['newspaper'].median()
    outlier_indices = advertising_data[(advertising_data['newspaper'] < lower_bound_advertising['newspaper']) | 
                                    (advertising_data['newspaper'] > upper_bound_advertising['newspaper'])].index

    advertising_data.loc[outlier_indices, 'newspaper'] = median_newspaper

    # Confirming that the outliers have been replaced
    outliers_after_replacement = advertising_data.loc[outlier_indices, 'newspaper']
    st.write(outliers_after_replacement)

    return advertising_data
