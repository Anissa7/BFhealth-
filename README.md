# Khelem Health Outcome Prediction

## Overview
This project aims to predict health outcomes using health indicators from Burkina Faso. The objective is to compare different machine learning approaches—including neural network models with various optimization techniques and a classical machine learning algorithm (Logistic Regression)—to determine the most effective method for accurate classification.

## Dataset
- **Source:** Health indicators dataset ("Khelem") from Burkina Faso.
- **Description:** The dataset contains a range of health metrics. For this project, we focus on the indicator `"NCD_BMI_30A"`. After extracting numeric values from the dataset, a binary target is created using the median value as the threshold.

## Models Implemented
- **Neural Network Models:**
  - **Baseline Model (Instance 1):** A simple NN model without any defined optimization techniques.
  - **Optimized Models (Instances 2 to 4):** These models incorporate optimization techniques such as different optimizers (Adam, RMSprop), early stopping, and regularization.
- **Classical ML Model (Instance 5):**
  - **Logistic Regression:** Tuned using GridSearchCV to optimize hyperparameters like the regularization strength and penalty (L1/L2).

## Experimentation
An experimentation table in the notebook compares the following aspects of each model:
- Training Instance
- Optimizer Used
- Regularizer Used
- Epochs
- Early Stopping (Yes/No)
- Number of Layers
- Learning Rate
- Performance Metrics: Accuracy, F1 Score, Recall, and Precision

Key findings and analysis of the performance differences are discussed in detail both in the notebook and in the video presentation.

## Instructions to Run the Notebook
1. **Clone the Repository:**
   ```bash
   git clone <repository-url>
   cd Project_Name
   ```

2. **Run the Notebook:**
   - Open `notebook.ipynb` in Jupyter Notebook or any compatible environment.
   - Execute the cells sequentially. The notebook is modularized into sections for data preprocessing, model training (Neural Networks and Logistic Regression), evaluation, and predictions.

3. **Loading a Saved Model for Predictions:**
   - Neural Network Example:
     ```python
     from tensorflow.keras.models import load_model
     best_model = load_model("my_model.keras")
     predictions = (best_model.predict(X_test) > 0.5).astype(int)
     print(predictions)
     ```
   - Logistic Regression Example:
     ```python
     import pickle
     best_lr_model = pickle.load(open("saved_models/finalized_logistic_model.sav", "rb"))
     predictions_lr = best_lr_model.predict(X_test)
     print(predictions_lr)
     ```

## Directory Structure
```
Project_Name/
├── notebook.ipynb
├── saved_models/
│   ├── finalized_model.sav
│   ├── finalized_model_2.sav
│   ├── finalized_model_3.sav
│   ├── finalized_model_4.sav
│   └── finalized_logistic_model.sav
└── README.md
```

## Video Presentation
A 5-minute video presentation accompanies this project submission. In the video, I:
- Explain the problem statement and dataset.
- Describe the different models implemented (both NN and Logistic Regression).
- Walk through the experimentation table and discuss which optimization techniques led to the best results.
- Provide instructions on how to run the notebook and load the saved models.

## Summary and Conclusions
The experimentation indicates that careful tuning—especially with the classical Logistic Regression model—can yield excellent performance. While the neural network models provide additional insights and flexibility in modeling, the Logistic Regression model (with tuned hyperparameters) achieved outstanding performance on the test set. Detailed error analysis and model comparisons are provided within the notebook.
