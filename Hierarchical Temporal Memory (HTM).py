import pandas as pd
import matplotlib.pyplot as plt
from htm.bindings.sdr import SDR
from htm.algorithms.temporal_memory import TemporalMemory
import numpy as np

def load_and_preprocess_data(file_path):
    """Load and preprocess healthcare IoT dataset."""
    df = pd.read_csv(file_path)

    # Drop missing values
    df.dropna(inplace=True)


    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df.set_index('Timestamp', inplace=True)


    df.rename(columns={
        'Temperature (°C)': 'Temperature',
        'Heart_Rate (bpm)': 'Heart_Rate'
    }, inplace=True)

    # Normalize data (0 to 1)
    df['Temperature'] = (df['Temperature'] - df['Temperature'].min()) / (df['Temperature'].max() - df['Temperature'].min())
    df['Heart_Rate'] = (df['Heart_Rate'] - df['Heart_Rate'].min()) / (df['Heart_Rate'].max() - df['Heart_Rate'].min())

    return df

def detect_anomalies(df):
    """Apply HTM for anomaly detection."""
    
    # Initialize HTM Temporal Memory
    tm = TemporalMemory(
        columnDimensions=(512,),
        cellsPerColumn=32,
        activationThreshold=12,
        initialPermanence=0.21,
        connectedPermanence=0.5,
        minThreshold=10,
        maxNewSynapseCount=20,
        permanenceIncrement=0.1,
        permanenceDecrement=0.1,
    )

    anomalies = []
    for row in df.itertuples(index=True, name=None):
        _, temp, hr = row  # Extract values

        # Create input SDR
        input_sdr = SDR((512,))
        input_sdr.randomize()  # Simple SDR encoding

        # Feed into HTM Temporal Memory
        tm.compute(input_sdr, learn=True)

        # Calculate anomaly score (simple fixed threshold)
        anomaly_score = 1 - (len(tm.getActiveCells()) / tm.getMaxCellsPerColumn())
        anomalies.append(-1 if anomaly_score > 0.75 else 1)

    df['Anomaly'] = anomalies

    return df

def plot_anomalies(df):
    """Plot temperature and heart rate with anomalies."""
    plt.figure(figsize=(12, 6))

    # Temperature Plot
    plt.subplot(2, 1, 1)
    plt.plot(df.index, df['Temperature'], label='Temperature', color='blue')
    plt.scatter(df[df['Anomaly'] == -1].index, df[df['Anomaly'] == -1]['Temperature'],
                color='red', label='Anomaly', marker='x')
    plt.xlabel("Timestamp")
    plt.ylabel("Temperature (Normalized)")
    plt.title("Temperature Trends with Anomalies")
    plt.legend()

    # Heart Rate Plot
    plt.subplot(2, 1, 2)
    plt.plot(df.index, df['Heart_Rate'], label='Heart Rate', color='green')
    plt.scatter(df[df['Anomaly'] == -1].index, df[df['Anomaly'] == -1]['Heart_Rate'],
                color='red', label='Anomaly', marker='x')
    plt.xlabel("Timestamp")
    plt.ylabel("Heart Rate (Normalized)")
    plt.title("Heart Rate Trends with Anomalies")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    dataset_path = r"D:\medical_gpt\Digital Twin\healthcare_iot_target_dataset.csv"
    
    # Load and process data
    df = load_and_preprocess_data(dataset_path)

    # Detect anomalies
    df = detect_anomalies(df)

    # Print results
    print("\nAnomalies Detected:")
    print(df[df['Anomaly'] == -1])  # View anomalies
    
    # Plot anomalies
    plot_anomalies(df)
