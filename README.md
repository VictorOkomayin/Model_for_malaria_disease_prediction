# Malaria Disease Prediction using Machine Learning
## Project Overview
This project applies machine learning techniques to predict malaria disease categories (Non-malaria Infection, Severe Malaria, Uncomplicated Malaria). The goal is to support early detection and improve healthcare outcomes by providing an automated prediction pipeline.

## Repository Structure
- Machine_Learning_Language_Model_for_Malaria_Disease_Prediction.ipynb → Main Jupyter Notebook with code, training, and evaluation.
- data/ → Dataset (link or instructions on how to obtain).
- models/ → Saved trained models (if applicable).
- README.md → Project documentation.

## Dataset
- Source: [Provide dataset link or citation here]
- Description: Contains patient features relevant to malaria diagnosis.
- Preprocessing:
- Standardization using MinMaxScaler (scaled between 0 and 1).
- Train-test split ratio: 80:20.

## Methodology
- Data Preprocessing
  - Feature scaling with MinMaxScaler.
  - Train-test split (80:20).
- Model Training
  - Algorithm: Random Forest Classifier.
  - Default hyperparameters used.
- Evaluation
  - Metrics: Balanced Accuracy, Precision, Recall, F1-score.
  - Confusion Matrix plotted for class-level performance.

## Results
The Random Forest classifier achieved the following performance on the test set:
- Balanced Accuracy: 79%
- F1-score (Weighted): 80%
- Precision (Weighted): 81%
- Recall (Weighted): 80.6%
- Confusion Matrix
  - The confusion matrix shows classification performance across the three classes:
    - Non-malaria Infection
    - Severe Malaria
    - Uncomplicated Malaria
- (Insert confusion matrix image here if saved from plot_confusion_matrix.)

## How to Run
- Clone the repository:
git clone https://github.com/VictorOkomayin/Model_for_malaria_disease_prediction.git
cd Model_for_malaria_disease_prediction
- Install dependencies:
pip install -r requirements.txt
- Open the notebook:
jupyter notebook Machine_Learning_Language_Model_for_Malaria_Disease_Prediction.ipynb

## Future Improvements
- Implement deep learning (CNNs) for image-based malaria detection.
- Add deployment pipeline (Flask/Streamlit app).
- Expand dataset for better generalization.
- Address ethical considerations (false negatives, patient safety).

## Contributing
Contributions are welcome! Please fork the repo and submit a pull request.

## License
This project is licensed under the MIT License — see the LICENSE file for details.

## Acknowledgements
- Dataset providers.
- Open-source libraries (Scikit-learn, Pandas, Matplotlib, etc.).
- Inspiration from malaria research initiatives.

