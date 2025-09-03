import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
import tensorflow as tf
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import train_test_split

matplotlib.use("TkAgg")


def data_explore():
    normal_df = pd.read_csv('ptbdb_normal.csv', header=None)
    abnormal_df = pd.read_csv('ptbdb_abnormal.csv', header=None)

    combined_df = pd.concat([normal_df, abnormal_df])

    class_counts = combined_df.iloc[:, -1].value_counts()
    print(class_counts)

    plt.figure(figsize=(8, 6))
    class_counts.plot(kind='bar')
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.xticks(ticks=[0, 1], labels=['Abnormal', 'Normal'], rotation=0)
    plt.show()

    labels = ['Normal', 'Abnormal']

    for class_label in combined_df.iloc[:, -1].unique():
        example = combined_df[combined_df.iloc[:, -1] == class_label].iloc[0,
                  :-1]
        plt.figure()
        plt.plot(example)
        plt.title(f'Example Time Series for {labels[int(class_label)]}')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.show()

    plt.figure()
    plt.plot(abnormal_df.iloc[8])
    plt.title(f'Example Time Series for Abnormal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()

    normal_data = normal_df.iloc[:, :-1].values
    abnormal_data = abnormal_df.iloc[:, :-1].values

    normal_mean = np.mean(normal_data, axis=0)
    normal_std = np.std(normal_data, axis=0)

    abnormal_mean = np.mean(abnormal_data, axis=0)
    abnormal_std = np.std(abnormal_data, axis=0)

    plt.figure(figsize=(10, 5))
    plt.plot(normal_mean, label='Mean')
    plt.fill_between(range(len(normal_mean)), normal_mean - normal_std, normal_mean + normal_std, alpha=0.2,
                     label='Standard Deviation')
    plt.title('Mean and Standard Deviation Over Time for Normal Data')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(abnormal_mean, label='Mean')
    plt.fill_between(range(len(abnormal_mean)), abnormal_mean - abnormal_std, abnormal_mean + abnormal_std, alpha=0.2,
                     label='Standard Deviation')
    plt.title('Mean and Standard Deviation Over Time for Abnormal Data')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()


def plot_loss(history):
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def plot_confusion_matrix1(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


def MLP_ECG_Diagnostic():
    df_normal = pd.read_csv('ptbdb_normal.csv', header=None)
    df_abnormal = pd.read_csv('ptbdb_abnormal.csv', header=None)

    data = pd.concat([df_normal, df_abnormal], axis=0).values
    X = data[:, :-1]
    y = data[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {i : class_weights[i] for i in range(len(class_weights))}

    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # history = model.fit(X_train, y_train, epochs=50, validation_split=0.1)
    history = model.fit(X_train, y_train, epochs=40, validation_split=0.1, class_weight=class_weights)

    plot_loss(history)

    y_pred = (model.predict(X_test) > 0.5).astype("int32")

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    report = classification_report(y_test, y_pred, output_dict=True)
    print(report)

    results = {
        "Configuration": ["MLP with 3 layers [64, 32, 16]"],
        "Accuracy": [accuracy],
        "Precision Normal": [report["0.0"]["precision"]],
        "Recall Normal": [report["0.0"]["recall"]],
        "F1-score Normal": [report["0.0"]["f1-score"]],
        "Precision Abnormal": [report["1.0"]["precision"]],
        "Recall Abnormal": [report["1.0"]["recall"]],
        "F1-score Abnormal": [report["1.0"]["f1-score"]]
    }

    results_df = pd.DataFrame(results)
    print(results_df)

    plot_confusion_matrix(y_test, y_pred)



def main():
    # data_explore()
    MLP_ECG_Diagnostic()
    # df = pd.read_csv("../Tema1/date_tema_1_iaut_2024.csv")
    # d, y = preprocessing(df)
    # print(d, y)
    # MLP_Custom_Dataset()


if __name__ == "__main__":
    main()
