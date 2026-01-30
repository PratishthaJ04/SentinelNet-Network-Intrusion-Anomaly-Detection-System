# SentinelNet-Network-Intrusion-Anomaly-Detection-System
An Unsupervised ML Framework for Real-Time Infrastructure Security 


Overview :
SentinelNet is a specialized Network Intrusion Detection System (NIDS) designed to identify malicious traffic patterns (DDoS, U2R, R2L, and Probing) using Isolation Forest. This project bridges the gap between infrastructure management and AI-driven security, applying unsupervised learning to detect "Zero-Day" threats without requiring labeled historical data for training.

This project is an extension of my research published in the Journal of Metrology Society of India, where I applied similar ML techniques to NTP server synchronization.


Technical Stack :
Language: Python 3.12
Machine Learning: Scikit-learn (Isolation Forest)
Data Processing: Pandas, NumPy
Visualization: Seaborn, Matplotlib
Infrastructure Context: Designed for integration with monitoring tools like Grafana and Zabbix.


Key Features :
Automated Feature Alignment: A robust preprocessing pipeline that synchronizes training and testing datasets, handling missing headers and categorical encoding automatically.
Unsupervised Detection: Uses an Isolation Forest algorithm to score the "anomaly degree" of network packets, making it highly effective for detecting unknown attack vectors.
Security Metrics: Evaluates performance using Precision, Recall, and F1-Score to ensure low false-positive rates in a production environment.


Results :
The model generates a Confusion Matrix (saved as detection_results.png) and a detailed Classification Report.
Normal Traffic Identification: High precision in identifying standard user behavior.
Anomaly Flagging: Effectively isolates spikes in protocol flags and connection durations that deviate from the baseline.

Conclusion : 
The SentinelNet: Network Intrusion & Anomaly Detection System was evaluated using a balanced test set of 4,000 connections (2,000 Normal / 2,000 Anomaly) to ensure statistical integrity and avoid the "Zero Support" bias found in unlabeled datasets. The system achieved a Normal Recall of 0.93, demonstrating high reliability in identifying legitimate traffic and preventing operational downtime—a critical factor in infrastructure monitoring environments like those I managed during my research at CSIR-NPL. While the Anomaly Recall of 0.13 and Precision of 0.65 highlight the inherent difficulty of using unsupervised Isolation Forest algorithms to detect subtle "Zero-Day" attacks that mimic baseline behavior, the results provide a transparent performance benchmark for anomaly detection in high-dimensional network data. By documenting these metrics alongside the generated Confusion Matrix, this project showcases a professional commitment to rigorous model evaluation, moving beyond simple accuracy to analyze the real-world trade-offs between security coverage and alert fatigue.
