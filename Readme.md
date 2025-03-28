# ANN-Based Classification and Regression with Streamlit Apps

This project demonstrates the use of Artificial Neural Networks (ANNs) for both classification and regression tasks. It leverages libraries such as TensorFlow and Keras for model building and training, and includes hyperparameter tuning using tools like `KerasClassifier` and `GridSearchCV`. Additionally, Streamlit apps have been created for both tasks to provide an interactive interface for predictions.

## Project Structure

```
ANN
├── classification/
│   ├── app.py # Streamlit app for classification
│   ├── Churn_Modelling.csv   # Dataset for classification
│   ├── experiments.ipynb # Notebook for classification experiments
│   ├── label_encoder_gender.pkl
│   ├── model.h5              # Trained classification model
│   ├── onehot_encoder_geo.pkl
│   ├── prediction.ipynb # Notebook for classification predictions
│   ├── scaler.pkl
├── regression/
│   ├── app.py    # Streamlit app for regression
│   ├── Churn_Modelling.csv   # Dataset for regression
│   ├── hyperparametertuningann.ipynb # Notebook for hyperparameter tuning
│   ├── label_encoder_gender.pkl
│   ├── onehot_encoder_geo.pkl
│   ├── salary_model.h5       # Trained regression model
│   ├── salaryregression.ipynb # Notebook for regression experiments
│   ├── scaler.pkl
├── requirements.txt 
├── .gitignore
```

## Features

### Classification
- Built an ANN model for classification tasks using TensorFlow and Keras.
- Preprocessed data using encoders (`LabelEncoder`, `OneHotEncoder`) and scalers.
- Saved the trained model and preprocessing artifacts for reuse.
- Created a Streamlit app to predict outcomes interactively.

### Regression
- Built an ANN model for regression tasks using TensorFlow and Keras.
- Performed hyperparameter tuning using `KerasClassifier` and `GridSearchCV` to find the optimal number of layers and neurons.
- Preprocessed data similarly to the classification task.
- Created a Streamlit app to predict salaries interactively.

## Installation

1. Clone the repository:
   ```bash
   https://github.com/MoB-369/ANN.git
   cd ANN
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies for both classification and regression:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Streamlit Apps

#### Classification App
Navigate to the `classification` folder and run the Streamlit app:
```bash
cd classification
streamlit run app.py
```

#### Regression App
Navigate to the `regression` folder and run the Streamlit app:
```bash
cd regression
streamlit run app.py
```

### Notebooks
- Use the Jupyter notebooks in the `classification` and `regression` folders to explore the experiments and hyperparameter tuning.

## Key Libraries Used
- **TensorFlow/Keras**: For building and training ANN models.
- **scikit-learn**: For preprocessing, hyperparameter tuning (`GridSearchCV`), and evaluation.
- **Streamlit**: For creating interactive web apps.
- **Pandas**: For data manipulation and analysis.

## Results
- **Classification**: Achieved high accuracy in predicting outcomes using the ANN model.
- **Regression**: Predicted salaries with minimal error after tuning the model's hyperparameters.

## Future Work
- Extend the models to handle more complex datasets.
- Add more advanced hyperparameter tuning techniques.
- Improve the Streamlit apps with additional features like visualizations.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
