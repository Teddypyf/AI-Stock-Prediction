import pandas as pd
import matplotlib.pyplot as plt

# Load forecast results and actual value data
prediction_df = pd.read_csv('predictions.csv')
actual_values = prediction_df['Actual'].values
predicted_values = prediction_df['Predicted'].values
predicted_Average = prediction_df['Rolling Average'].values
diffrens = abs(prediction_df['Rolling Average'].values - prediction_df['Actual'].values)

# Prompts the user to select what to draw
print("Choose the data to plot (comma-separated numbers):")
print("1. Actual Values")
print("2. Predicted Values")
print("3. Predicted Average")
print("4. Difference (Predicted Average - Actual)")

choices = input("Enter your choices separated by comma (e.g., 1,3): ").split(',')
choices = [int(choice.strip()) for choice in choices]

# Create a new figure
plt.figure(figsize=(12, 7.5))

# Plot data based on user selections
for choice in choices:
    if choice == 1:
        plt.plot(actual_values, label='Actual', color='blue', linestyle='-')
    elif choice == 2:
        plt.plot(predicted_values, label='Predicted', color='green', linestyle='--')
    elif choice == 3:
        plt.plot(predicted_Average, label='Predicted Average', color='red', linestyle='-')
    elif choice == 4:
        plt.plot(diffrens, label='Difference', color='black', linestyle='-')
    else:
        print(f"Ignoring invalid choice: {choice}")

plt.legend()

# Add tags and titles
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Selected Data Plot')

# display figure
plt.show()
