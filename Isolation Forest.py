import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

def load_and_preprocess_data(file_path):
    """Load and preprocess healthcare IoT dataset."""
    df = pd.read_csv(file_path)

    # Debug: Print column names
    print("Columns in dataset:", df.columns)

    # Drop missing values
    df.dropna(inplace=True)

    # Convert timestamp to datetime & keep it as index
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df.set_index('Timestamp', inplace=True)
    else:
        print("Warning: 'Timestamp' column not found.")

    # Rename columns if they exist
    column_mapping = {
        'Temperature (°C)': 'Temperature',
        'Heart_Rate (bpm)': 'Heart_Rate',
        'Target_Heart_Rate': 'Target_Heart_Rate'
    }
    df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns}, inplace=True)

    # Ensure required columns exist before training
    required_cols = ['Temperature', 'Heart_Rate', 'Target_Heart_Rate']
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing expected column: {col}")

    # Train anomaly detection model
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(df[required_cols])

    # Predict anomalies
    df['Anomaly'] = model.predict(df[required_cols])

    return df

if __name__ == "__main__":
    df = load_and_preprocess_data("healthcare_iot_target_dataset.csv")
    print(df[df['Anomaly'] == -1])  # View abnormal patients


def plot_anomalies(df):
    """Plot temperature and heart rate with anomalies."""
    plt.figure(figsize=(12, 6))

    # Temperature Plot
    plt.subplot(2, 1, 1)
    plt.plot(df.index, df['Temperature'], label='Temperature', color='blue')
    plt.scatter(df[df['Anomaly'] == -1].index, df[df['Anomaly'] == -1]['Temperature'],
                color='red', label='Anomaly', marker='x')
    plt.xlabel("Timestamp")
    plt.ylabel("Temperature (°C)")
    plt.title("Temperature Trends with Anomalies")
    plt.legend()

    # Heart Rate Plot
    plt.subplot(2, 1, 2)
    plt.plot(df.index, df['Heart_Rate'], label='Heart Rate', color='green')
    plt.scatter(df[df['Anomaly'] == -1].index, df[df['Anomaly'] == -1]['Heart_Rate'],
                color='red', label='Anomaly', marker='x')
    plt.xlabel("Timestamp")
    plt.ylabel("Heart Rate (bpm)")
    plt.title("Heart Rate Trends with Anomalies")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = load_and_preprocess_data("healthcare_iot_target_dataset.csv")
    print(df[df['Anomaly'] == -1])  # View anomalies
    plot_anomalies(df)  # Visualize the anomalies