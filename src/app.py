import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

import subprocess

subprocess.run(["pip", "install", "streamlit", "--upgrade"], check=True)

import streamlit as st
from matplotlib.figure import Figure

import pyspark_func as pyspark_func
import scikit_func as scikit_func

import threading

_lock = threading.Lock()

st.set_page_config(
    page_title='Streamlit with Healthcare Data',
    layout="wide",
    initial_sidebar_state="expanded",
)


def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    elif clf_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
    elif clf_name == 'Decision Tree':
        params['criterion'] = st.sidebar.radio("criterion", ('gini', 'entropy'))
        params['max_features'] = st.sidebar.selectbox("max_features", (None, 'auto', 'sqrt', 'log2'))
        params['max_depth'] = st.sidebar.slider('max_depth', 1, 32)
        params['min_samples_split'] = st.sidebar.slider('min_samples_split', 0.1, 1.0)
    return params


def pyspark_buildmodel(pyspark_classifier_name):
    spark = pyspark_func.get_spark_session()
    trainingData, testData = pyspark_func.prepare_dataset(spark, data)
    return pyspark_func.training(spark, pyspark_classifier_name, trainingData, testData)


def pyspark_operation(pyspark_col):
    st.sidebar.subheader('PySpark')
    pyspark_classifier_name = st.sidebar.selectbox(
        'Select classifier',
        pyspark_func.get_sidebar_classifier(), key='pyspark'
    )
    pyspark_col.write(f'Classifier = {pyspark_classifier_name}')
    accuracy = pyspark_buildmodel(pyspark_classifier_name)
    pyspark_col.write(f'Accuracy = {accuracy}')


def create_sidelayout(scipy_col, pyspark_col):
    """
    Create machine learning options and handle plots
    :param scipy_col: Scikit-Learn section in the main page
    :param pyspark_col: PySpark section in the main page
    """

    st.sidebar.title('Machine Learning Options')

    # Select the classifier and handle parameter selection
    scikit_classifier_name = st.sidebar.selectbox(
        'Select classifier',
        scikit_func.get_sidebar_classifier(),
        key='scikit'
    )

    # Display classifier information in the Scikit-Learn section
    scipy_col.write(f'Classifier = {scikit_classifier_name}')

    # Handle parameter selection for the selected classifier
    params = add_parameter_ui(scikit_classifier_name)

    # Train the model and calculate accuracy
    accuracy = scikit_func.trigger_classifier(scikit_classifier_name, params, X_train, X_test, y_train, y_test)

    # Display accuracy in the Scikit-Learn section
    scipy_col.write(f'Accuracy = {accuracy}')

    # Check if PySpark is enabled and handle PySpark operations if so
    if pyspark_enabled == 'Yes':
        pyspark_operation(pyspark_col)

        # Check if the selected classifier is supported for plotting in PySpark
        if scikit_func.get_sidebar_classifier() in ['Decision Tree', 'Random Forest']:
            # Plot the corresponding PySpark plots
            for i in range(3):
                plot(scipy_col, pyspark_col, i)
        else:
            # Display a warning for PySpark plotting
            st.warning('Plots for PySpark are not available yet')



def create_subcol(data, X_test, y_test, model):
    """
    Create multiple columns based on the number of plots
    :return: Tuple containing Streamlit columns
    """
    num_plots = 3  # Assuming you have 3 groups of plots
    fig_size = (8, 6)  # Assuming a default figure size

    # Create Streamlit columns
    scipy_col, pyspark_col = st.columns(2)

    # Display the Scikit-Learn section
    scipy_col.header('''
        __Scikit Learn ML__
        ![scikit](https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png)''')

    # Dynamically create columns for the Scikit-Learn plots
    for i in range(num_plots):
        col_plot = scipy_col.columns(2)  # Use 'scipy_col' instead of 'st'
        with col_plot[0], _lock:
            st.subheader(f'Scikit-Learn Plot {i + 1}')
        with col_plot[1]:
            # Render the corresponding plot within the column
            render_plot(scipy_col, pyspark_col, i, f'Scikit-Learn Plot {i + 1}', fig_size, data, X_test, y_test, model)

    # Check if PySpark is enabled and display the PySpark section if so
    if pyspark_enabled == 'Yes':
        pyspark_col.header('''
              __PySpark Learn ML__
              ![pyspark](https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/Apache_Spark_logo.svg/120px-Apache_Spark_logo.svg.png)
              ''')

        # Dynamically create columns for the PySpark plots
        for i in range(num_plots):
            col_plot = pyspark_col.columns(2)  # Use 'pyspark_col' instead of 'st'
            with col_plot[0], _lock:
                st.subheader(f'PySpark Plot {i + 1}')
            with col_plot[1]:
                # Render the corresponding plot within the column
                render_plot(scipy_col, pyspark_col, num_plots + i, f'PySpark Plot {i + 1}', fig_size, data, X_test, y_test, model)

    return scipy_col, pyspark_col

    
def render_plot(scipy_col, pyspark_col, i, plot_title, fig_size, data, X, y_test, model):
    # Create the figure and axes
    fig, ax = plt.subplots(figsize=fig_size)

    # Generate the appropriate plot based on the plot group index
    if 0 <= i <= 2:
        if i == 0:
            plt.hist(data['cardio'], edgecolor='black', bins=10, alpha=0.7)
            ax.set_title('Histogram of Cardio Labels')
            ax.set_xlabel('Cardio Label')
            ax.set_ylabel('Frequency')
        elif i == 1:
            # Assuming your data is stored in a DataFrame named 'data'
            categorical_columns = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio']
            
            # Select only the categorical columns from the DataFrame
            categorical_data = data[categorical_columns]
            
            # Melt the DataFrame to create df_categorical
            df_categorical = pd.melt(categorical_data)
            
            # Now df_categorical is ready to be used in the sns.countplot or other analyses
            sns.countplot(
                x="variable", hue="value", data=df_categorical, ax=ax
            )
            ax.set_title('Count Plot of Categorical Variables')
            ax.set_xlabel('Variable')
            ax.set_ylabel('Count')
        elif i == 2:
            # Handle feature importances based on the classifier type
            if hasattr(model, 'feature_importances_'):
                feature_importances = model.feature_importances_
                ax.bar(X.columns, feature_importances)
                ax.set_title('Feature Importances')
                ax.set_xlabel('Feature')
                ax.set_ylabel('Feature Importance')
    elif 3 <= i <= 4:
        if i == 3:
            # ROC/AUC Plot - Execute only for i == 3
            # Assuming your model has a predict_proba method
            if hasattr(model, 'predict_proba'):
                fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X)[:, 1])
                ax.plot(fpr, tpr)
                ax.set_title('ROC Curve')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                st.write('AUC:', round(roc_auc_score(y_test, model.predict_proba(X)[:, 1]), 3))
        elif i == 4:
            # Assuming your model has a predict method
            if hasattr(model, 'predict'):
                confusion_mat = confusion_matrix(y_test, model.predict(X))
                sns.heatmap(
                    confusion_mat, annot=True, fmt='g', linewidths=0.5, ax=ax
                )
                ax.set_title('Confusion Matrix')
                ax.set_xlabel('Predicted Label')
                ax.set_ylabel('Actual Label')

    # Display the plot in the appropriate column
    scipy_col.pyplot(fig)

    # Return the figure and axes
    return fig, ax
    

# Load data and prepare dataset
data = scikit_func.load_data()
X_train, X_test, y_train, y_test = scikit_func.prepare_dataset(data)

# Sidebar controls
st.sidebar.markdown('''
__⛔ PySpark is a computational expensive operation __  
Selecting _Yes_ will trigger Spark Session automatically!  
''', unsafe_allow_html=True)
pyspark_enabled = st.sidebar.radio("PySpark_Enabled", ('No', 'Yes'))


def main():
    st.title('''Streamlit ![](https://assets.website-files.com/5dc3b47ddc6c0c2a1af74ad0/5e0a328bedb754beb8a973f9_logomark_website.png) Healthcare ML Data App''')
    st.subheader('Streamlit Healthcare example By [Enrique Real](https://es.linkedin.com/in/enrique-real-bru-057150237)')
    st.markdown("**Cardiovascular Disease dataset by Kaggle** 👇")
    st.markdown('''
    This is the source of the dataset and the problem statement [Kaggle Link](https://www.kaggle.com/sulianova/cardiovascular-disease-dataset)  
    The dataset consists of 70 000 records of patients data, 11 features + target  
    
    ◀️ Running the same data over ___Scikit Learn & Pyspark ML___ - A simple Comparison and Selection  
    ___This is just for demonstration & absolutely not about the Best Machine Learning Model!___  
    ''')

    st.dataframe(data=data.head(20), height=200)
    st.write('Shape of dataset:', X_train.shape)
    st.write('number of classes:', len(np.unique(y_test)))

    # Create columns for Scikit-learn and PySpark
    scikit_col, pyspark_col = st.columns(2)

    # Initialize the plots list outside both loops
    plots = []

    # Check if PySpark is enabled
    if pyspark_enabled == 'Yes':
        # Use PySpark model
        pyspark_col.header('''
              __PySpark Learn ML__
              ![pyspark](https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/Apache_Spark_logo.svg/120px-Apache_Spark_logo.svg.png)
              ''')

        # Render the general plots for PySpark
        for i in range(3):  # Assuming there are 3 general plots
            fig, ax = render_plot(scikit_col, pyspark_col, i, f'General Plot {i + 1}', (8, 6), data, X_test, y_test, None)
            plots.append((fig, ax))

        # Dynamically create columns for the PySpark plots
        for i in range(2):  # Assuming there are 2 PySpark-specific plots
            col_plot = pyspark_col.columns(2)
            with col_plot[0], _lock:
                st.subheader(f'PySpark Plot {i + 1}')
            with col_plot[1]:
                # Render the corresponding plot within the column
                classifier_name = pyspark_col.selectbox(
                    'Select classifier',
                    pyspark_func.get_sidebar_classifier(),
                    key='pyspark'
                )
                params = add_parameter_ui(classifier_name)
                pyspark_col.write(f'Classifier = {classifier_name}')
                pyspark_col.write(f'Accuracy = {pyspark_func.trigger_classifier(classifier_name, params, X_train, X_test, y_train, y_test)}')

                # Instantiate the PySpark model based on the selected classifier
                model = pyspark_func.get_model(classifier_name, params)

                # Render the plots using the render_plot function
                for j in range(2):  # Assuming there are 2 PySpark-specific plots
                    fig, ax = render_plot(scikit_col, pyspark_col, j + 3, f'PySpark Plot {j + 1}', (8, 6), data, X_test, y_test, model)
                    plots.append((fig, ax))

        # Display the plots for PySpark
        for fig, ax in plots:
            pyspark_col.pyplot(fig)

    else:
        # Use Scikit-learn model
        scikit_col.header('''
            __Scikit Learn ML__
            ![scikit](https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png)''')

        # Render the general plots for Scikit-Learn
        for i in range(3):  # Assuming there are 3 general plots
            fig, ax = render_plot(scikit_col, pyspark_col, i, f'General Plot {i + 1}', (8, 6), data, X_test, y_test, None)
            plots.append((fig, ax))

        # Dynamically create columns for the Scikit-Learn plots
        for i in range(2):  # Assuming there are 2 Scikit-Learn-specific plots
            col_plot = scikit_col.columns(2)
            plots = []  # Store the plots
            with col_plot[0], _lock:
                st.subheader(f'Scikit-Learn Plot {i + 1}')
            with col_plot[1]:
                # Render the corresponding plot within the column
                classifier_name = scikit_col.selectbox(
                    'Select classifier',
                    scikit_func.get_sidebar_classifier(),
                    key='scikit'
                )
                params = add_parameter_ui(classifier_name)
                scikit_col.write(f'Classifier = {classifier_name}')
                scikit_col.write(f'Accuracy = {scikit_func.trigger_classifier(classifier_name, params, X_train, X_test, y_train, y_test)}')

                # Instantiate the Scikit-learn model based on the selected classifier
                model = scikit_func.get_model(classifier_name, params)

                # Render the plots using the render_plot function
                for j in range(2):  # Assuming there are 2 Scikit-Learn-specific plots
                    fig, ax = render_plot(scikit_col, pyspark_col, j + 3, f'Scikit-Learn Plot {j + 1}', (8, 6), data, X_test, y_test, model)
                    plots.append((fig, ax))

                # Display the plots
                for fig, ax in plots:
                    scikit_col.pyplot(fig)

    # Display the layout for Scikit Learn and PySpark
    create_sidelayout(scikit_col, pyspark_col)

if __name__ == '__main__':
    main()
