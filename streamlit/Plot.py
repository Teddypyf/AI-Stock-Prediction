import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

def main():
    st.title("Data Plotter")

    # Load forecast results and actual value data
    prediction_df = pd.read_csv('predictions.csv')
    actual_values = prediction_df['Actual'].values
    predicted_values = prediction_df['Predicted'].values
    predicted_Average = prediction_df['Rolling Average'].values
    diffrens = abs(prediction_df['Rolling Average'].values - prediction_df['Actual'].values)

    # Prompts the user to select what to draw
    st.sidebar.markdown("### Choose the data to plot")
    st.sidebar.markdown("Select the data you want to plot from the options below:")
    actual_check = st.sidebar.checkbox("Actual Values")
    predicted_check = st.sidebar.checkbox("Predicted Values")
    average_check = st.sidebar.checkbox("Predicted Average")
    difference_check = st.sidebar.checkbox("Difference (Predicted Average - Actual)")

    if actual_check:
        plt.plot(actual_values, label='Actual', color='blue', linestyle='-')
    if predicted_check:
        plt.plot(predicted_values, label='Predicted', color='green', linestyle='--')
    if average_check:
        plt.plot(predicted_Average, label='Predicted Average', color='red', linestyle='-')
    if difference_check:
        plt.plot(diffrens, label='Difference', color='black', linestyle='-')

    plt.legend()

    # Add tags and titles
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Selected Data Plot')

    # display figure
    st.pyplot(plt)

if __name__ == "__main__":
    main()
