# Network Anomaly Detection Using Deep Learning

## Project Overview
This project implements a deep learning framework for network anomaly detection using the NSL-KDD dataset. The objective is to develop a robust model capable of identifying anomalous network activities, such as intrusions or cyberattacks, by leveraging deep learning techniques. The solution is designed to support cybersecurity professionals and researchers in enhancing network security through data-driven anomaly detection.

The project employs a dense neural network architecture built with TensorFlow to classify network traffic as normal or anomalous. Implemented in Python on Google Colab, the project utilizes a suite of data science libraries to ensure an efficient, reproducible, and scalable workflow. Interactive visualizations and comprehensive documentation are provided to facilitate result interpretation and replication.

## Key Features
- **Anomaly Detection**: Utilizes a dense neural network to classify network traffic as normal or anomalous.
- **Data Analysis**: Performs exploratory data analysis and preprocessing using Pandas and NumPy.
- **Model Training**: Implements and trains a deep learning model using TensorFlow and Scikit-learn for evaluation.
- **Visualization**: Generates visualizations to analyze data patterns and model performance.
- **Reproducibility**: Provides complete source code and documentation for result replication.

## Tools and Technologies
The project is implemented in Python on Google Colab, utilizing the following libraries:
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computations and matrix operations.
- **Scikit-learn**: For data preprocessing and model evaluation metrics.
- **TensorFlow**: For building and training the dense neural network model.

## Dataset
The project uses the **NSL-KDD dataset**, a widely recognized benchmark dataset for network intrusion detection. The dataset includes features such as:
- Network connection attributes (e.g., duration, protocol type, service).
- Traffic statistics (e.g., byte counts, packet rates).
- Labels indicating normal or anomalous (attack) traffic, with categories such as DoS, Probe, R2L, and U2R.

**Data Source**: The NSL-KDD dataset is publicly available and can be downloaded from [https://www.unb.ca/cic/datasets/nsl.html](https://www.unb.ca/cic/datasets/nsl.html).

**Dataset Storage**:
- Sample dataset files are included in the `/data` directory for convenience.
- The full NSL-KDD dataset is available via the provided link above. Due to size constraints, larger dataset files are not stored in the repository but can be downloaded and placed in the `/data` directory.

## Model Architecture
The model is a **dense neural network** implemented in TensorFlow, consisting of:
- Multiple fully connected (dense) layers with ReLU activation functions.
- A softmax output layer for multi-class classification (normal vs. various attack types).
- Optimization using the Adam optimizer and categorical cross-entropy loss.
- Regularization techniques (e.g., dropout) to prevent overfitting, if applicable.

Model performance is evaluated using metrics such as accuracy, precision, recall, F1-score, and confusion matrix, implemented via Scikit-learn.
