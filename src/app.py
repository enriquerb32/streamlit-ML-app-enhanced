import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

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


def create_subcol():
    """
    Create multiple columns based on the number of plots
    :return:
    """
    num_plots = 3  # Assuming you have 3 groups of plots
    scipy_col, pyspark_col = st.columns(2)

    # Display the Scikit-Learn section
    scipy_col.header('''
        __Scikit Learn ML__
        ![scikit](https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png)''')

    # Dynamically create columns for the Scikit-Learn plots
    for i in range(num_plots):
        col_plot = st.columns(2)
        with col_plot[0], _lock:
            st.subheader(f'Scikit-Learn Plot {i + 1}')
        with col_plot[1]:
            # Render the corresponding plot within the column
            fig_size = (8, 6)
            render_plot(scipy_col, pyspark_col, i, f'Scikit-Learn Plot {i + 1}', fig_size)

    # Check if PySpark is enabled and display the PySpark section if so
    if pyspark_enabled == 'Yes':
        pyspark_col.header('''
              __PySpark Learn ML__
              ![pyspark](https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/Apache_Spark_logo.svg/120px-Apache_Spark_logo.svg.png)
              ''')

        # Dynamically create columns for the PySpark plots
        for i in range(num_plots):
            col_plot = st.columns(2)
            with col_plot[0], _lock:
                st.subheader(f'PySpark Plot {i + 1}')
            with col_plot[1]:
                # Render the corresponding plot within the column
                render_plot(scipy_col, pyspark_col, num_plots + i, f'PySpark Plot {i + 1}')

    return scipy_col, pyspark_col

def plot(scipy_col, pyspark_col, i, scikit_classifier_name):  
    if i == 0:
        plot_title = "Cardiovascular Diseases Distribution"
        fig_size = (6, 4)
        col_plot = st.columns(2)
    elif i == 1:
        plot_title = "People Exposed to CVD more"
        fig_size = (6, 4)
        col_plot = st.columns(2)
    elif i == 2:
        if scikit_classifier_name == "Decision Tree":
            plot_title = "Feature Importances - Scikit-Learn"
        elif scikit_classifier_name == "Random Forest":
            plot_title = "Feature Importances - Scikit-Learn"
        else:
            plot_title = "Feature Importances - PySpark"
        fig_size = (10, 6)
        col_plot = st.columns(3)
    elif i == 3:
        plot_title = "ROC Curve"
        fig_size = (6, 4)
        col_plot = st.columns(2)
    elif i == 4:
        plot_title = "Confusion Matrix"
        fig_size = (7, 5)
        col_plot = st.columns(2)

    # Create a new thread to render the plot
    plot_thread = threading.Thread(target=_render_plot, args=(scipy_col, pyspark_col, i, plot_title, fig_size))
    plot_thread.start()
    

def render_plot(scipy_col, pyspark_col, i, plot_title, fig_size):
    # Create the figure and axes
    fig, ax = plt.subplots(figsize=fig_size)

    # Generate the appropriate plot based on the plot group index
    if i == 0:
        plt.hist(data['cardio'], edgecolor='black', bins=10, alpha=0.7)
    elif i == 1:
        sns.countplot(
            x="variable", hue="value", data=pd.melt(df_categorical), ax=ax
        )
    elif i == 2:
        feature_importances = model.get_feature_importances()
        ax.bar(X.columns, feature_importances)
        ax.set_xlabel('Feature')
        ax.set_ylabel('Feature Importance')
    elif i == 3:
        fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        ax.plot(fpr, tpr)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.set_title('ROC Curve')
        st.write('AUC:', round(roc_auc, 3))
    elif i == 4:
        confusion_matrix = confusion_matrix(y_test, model.predict(X_test))
        sns.heatmap(
            confusion_matrix, annot=True, fmt='g', linewidths=0.5, ax=ax
        )
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('Actual Label')

    # Notify the main thread that the plot has been rendered
    with _lock:
        st.pyplot(fig)


data = scikit_func.load_data()
X_train, X_test, y_train, y_test = scikit_func.prepare_dataset(data)
st.sidebar.markdown('''
__⛔ PySpark is computational expensive operation __  
Selecting _Yes_ will trigger Spark Session automatically!  
''', unsafe_allow_html=True)
pyspark_enabled = st.sidebar.radio("PySpark_Enabled", ('No', 'Yes'))


def main():
    st.title(
        '''Streamlit ![](https://assets.website-files.com/5dc3b47ddc6c0c2a1af74ad0/5e0a328bedb754beb8a973f9_logomark_website.png) Healthcare ML Data App''')
    st.subheader(
        'Streamlit Healthcare example By '
        '[Enrique Real](https://es.linkedin.com/in/enrique-real-bru-057150237)')
    st.markdown(
        "**Cardiovascular Disease dataset by Kaggle** 👇"
    )
    st.markdown('''
    This is source of the dataset and problem statement [Kaggle Link](https://www.kaggle.com/sulianova/cardiovascular-disease-dataset)  
    The dataset consists of 70 000 records of patients data, 11 features + target  
    
    ◀️ Running the same data over ___Scikit Learn & Pyspark ML___ - A simple Comparison and Selection  
    ___This is just for demonstration & absolutely not about Best Machine Learning Model!___  
    ''')
    st.dataframe(data=data.head(20), height=200)
    st.write('Shape of dataset:', X_train.shape)
    st.write('number of classes:', len(np.unique(y_test)))
    scipy_col, pyspark_col = create_subcol()
    create_sidelayout(scipy_col, pyspark_col)
    plot()


if __name__ == '__main__':
    main()
