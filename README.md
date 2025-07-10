# Anomaly-Detection-on-Network-Traffic-using-KNN-and-LSTM

This Python script implements an end-to-end pipeline for detecting anomalies (potential attacks) in network traffic data using:

K-Nearest Neighbors (KNN)

Bidirectional LSTM neural network

It loads CSV data (e.g., CICIDS dataset), preprocesses it, trains models, evaluates performance, visualizes predictions, and exports detected anomalies.

✨ Features
✅ Data Preprocessing

Loads CSV traffic data

Selects numeric columns

Scales features using MinMaxScaler

Creates sequences for time-series modeling

✅ KNN Classifier

Predicts anomalies based on feature similarity

✅ Bidirectional LSTM Neural Network

Learns temporal patterns for anomaly prediction

✅ Evaluation Metrics

Accuracy and F1 Score

✅ Visualization

Plots actual vs predicted anomalies

✅ Export

Saves detected anomalies to a CSV file

🛠️ Requirements
Python 3.x

pandas

numpy

scikit-learn

matplotlib

tensorflow

Install dependencies:

bash
Copy
Edit
pip install pandas numpy scikit-learn matplotlib tensorflow
📂 Dataset
Make sure your dataset CSV file exists at:

swift
Copy
Edit
/home/kartik/dti/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
You can change the file path by editing:

python
Copy
Edit
csv_file_path = r"/path/to/your/data.csv"

🚀 Usage

The script will:

Load and preprocess the dataset.

Train KNN and LSTM models.

Evaluate accuracy and F1-Score.

Display a graph comparing actual and predicted anomalies.

Save anomalies to:
