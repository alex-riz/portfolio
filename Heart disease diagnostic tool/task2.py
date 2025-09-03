import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.optimizers import Adam, SGD

matplotlib.use("TkAgg")

def discreteToNumerical(df):
    mapping = {'Automobile': 1, 'Bike': 2, 'Motorbike': 3, 'Public_Transportation': 4, 'Walking': 5, 'no': 0, 'yes': 1, 'Always': 3, 'Frequently': 2, 'Sometimes': 1, 'Female': 11, 'Male': 10}
    return df.replace(mapping)

def preprocessing(df):
    numerical_attributes = ['Regular_fiber_diet', 'Sedentary_hours_daily', 'Age', 'Est_avg_calorie_intake',
                            'Main_meals_daily', 'Height', 'Water_daily',
                            'Weight', 'Technology_time_use']

    discrete_attributes = ['Transportation', 'Diagnostic_in_family_history', 'High_calorie_diet', 'Alcohol', 'Snacks', 'Smoker', 'Calorie_monitoring', 'Gender']

    df[numerical_attributes] = df[numerical_attributes].replace(',', '.', regex=True)
    df[numerical_attributes] = df[numerical_attributes].astype(float)

    condition = ((df['Sedentary_hours_daily'] < 24) & (df['Age'] < 100) & (df['Height'] < 3) & (df['Weight'] < 200) & (df['Technology_time_use'] < 24) & (df['Regular_fiber_diet'] < 100))
    filtered_df = df[condition]

    X_train_numeric = filtered_df[numerical_attributes].values
    X_train_discrete = discreteToNumerical(filtered_df[discrete_attributes]).values
    y_train = filtered_df['Diagnostic'].values

    imp = SimpleImputer(missing_values=-1.0, strategy='mean')
    X_train_numeric_imp = imp.fit_transform(X_train_numeric)

    scaler = StandardScaler()
    X_train_numeric_std = scaler.fit_transform(X_train_numeric_imp)

    selector_numeric = SelectPercentile(percentile=40)
    X_train_numeric_selected = selector_numeric.fit_transform(X_train_numeric_std, y_train)

    selector_categorical = SelectPercentile(score_func=chi2, percentile=40)
    X_train_categorical_selected = selector_categorical.fit_transform(X_train_discrete, y_train)
    print(X_train_categorical_selected, X_train_numeric_selected)

    final_discrete = filtered_df[['Transportation', 'Diagnostic_in_family_history', 'Snacks', 'Calorie_monitoring']].values
    encoder = OneHotEncoder()
    encoded_discrete = encoder.fit_transform(final_discrete).toarray()

    final_numeric = filtered_df[['Age', 'Est_avg_calorie_intake', 'Main_meals_daily', 'Water_daily']].values
    std_numeric = scaler.fit_transform(final_numeric)
    concatenated_df = np.concatenate([encoded_discrete, std_numeric], axis=1)

    return concatenated_df, y_train

df = pd.read_csv("../Tema1/date_tema_1_iaut_2024.csv")
X, y = preprocessing(df)

d1 = pd.DataFrame(X)
d1.to_csv("ceseve.csv", index=False)
print(d1.head())
print(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
label_mapping = {'D0': 0, 'D1': 1, 'D2': 2, 'D3': 3, 'D4': 4, 'D5': 5, 'D6': 6}

y_train_numerical = np.array([label_mapping[label] for label in y_train])
y_test_numerical = np.array([label_mapping[label] for label in y_test])

y_train_encoded = to_categorical(y_train_numerical, num_classes=7)
y_test_encoded = to_categorical(y_test_numerical, num_classes=7)
optimizer = SGD(learning_rate=0.01, momentum=0.9)

model = Sequential()
model.add(Dense(187,activation='relu' , input_shape=(X_train.shape[1],)))
model.add(Dense(100 , activation='relu'))
model.add(Dense(7,activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train_encoded, epochs=500, batch_size=32, validation_data=(X_test, y_test_encoded))

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

print("Classification Report:")
print(classification_report(y_test_numerical, y_pred_classes))

conf_matrix = confusion_matrix(y_test_numerical, y_pred_classes)
print("Confusion Matrix:")
print(conf_matrix)

plt.figure(figsize=(10, 7))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(7)
plt.xticks(tick_marks, label_mapping.keys(), rotation=45)
plt.yticks(tick_marks, label_mapping.keys())

thresh = conf_matrix.max() / 2.
for i, j in np.ndindex(conf_matrix.shape):
    plt.text(j, i, format(conf_matrix[i, j], 'd'),
             horizontalalignment="center",
             color="white" if conf_matrix[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

print("Confusion Matrix:")
print(confusion_matrix(y_test_numerical, y_pred_classes))

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
