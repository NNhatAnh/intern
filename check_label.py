import pandas as pd

file_path = './dataset_intrument.csv'

try:
    df = pd.read_csv(file_path, sep=';')

    value_counts = df['label'].value_counts()

    total_count = len(df['label'])

    for value, count in value_counts.items():
        percentage = (count / total_count) * 100
        print(f"{value}: {count} entries ({percentage:.2f}%)")

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except KeyError:
    print(f"Error: The 'label' column was not found in the CSV file.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")