from glob import glob
import os
import numpy as np
import cv2
from time import time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

#################################################################################################

# Specify the path to your dataset
dataset_path = "./training"

# Read images and labels from the dataset
images = []
labels = []

# Define the desired dimensions for your images
desired_width = 100
desired_height = 100

#################################################################################################

for person_id, person_folder in enumerate(sorted(glob(os.path.join(dataset_path, "*")))):
    person_name = os.path.basename(person_folder)
    for image_path in glob(os.path.join(person_folder, "*.jpg")):
        # Read the image using OpenCV
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # Resize the image to the desired dimensions
        image = cv2.resize(image, (desired_width, desired_height))
        # Flatten the image and add it to the list
        images.append(image.flatten())
        # Extract the person's name from the filename and add it to the labels
        person_name_in_filename = os.path.splitext(os.path.basename(image_path))[0].split("_")[0]
        labels.append(person_name_in_filename)

#################################################################################################

# Convert the lists to numpy arrays
X = np.array(images)
y = np.array(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# #############################################################################
# Compute PCA (eigenfaces) on the face dataset: unsupervised feature extraction

n_components = 150

print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X_train.shape[0]))
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))

eigenfaces = pca.components_.reshape((n_components, desired_height, desired_width))

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))

#################################################################################################


# #############################################################################
# Train a SVM classification model

print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(
    SVC(kernel='rbf', class_weight='balanced'), param_grid
)
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)

# #############################################################################
# Quantitative evaluation of the model quality on the test set

print("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=np.unique(y)))

# Print confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=np.unique(y))
print("Confusion Matrix:")
print(conf_matrix)

# Calculate and print accuracy, precision, recall, and F1-score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# #############################################################################
# Implement CNN Model

# Reshape the flattened images to 2D arrays
X_train_reshaped = X_train.reshape(-1, desired_height, desired_width, 1)
X_test_reshaped = X_test.reshape(-1, desired_height, desired_width, 1)

# Convert labels to numerical values
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Normalize pixel values to be between 0 and 1
X_train_normalized = X_train_reshaped / 255.0
X_test_normalized = X_test_reshaped / 255.0

X_train_cnn = X_train_normalized
X_val_cnn = X_test_normalized
y_train_cnn = y_train_encoded
y_val_cnn = y_test_encoded

######## You have to know what the following code does##########################################################
# Define the CNN model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(desired_height, desired_width, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(np.unique(y)), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Early stopping callback to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the CNN model
history = model.fit(X_train_cnn, y_train_cnn, epochs=20, validation_data=(X_val_cnn, y_val_cnn), callbacks=[early_stopping])
################################################################################################################
# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test_normalized, y_test_encoded)
print(f'Test Accuracy (CNN): {test_accuracy:.2f}')

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Evaluate the CNN model on the test set
y_pred_cnn_probs = model.predict(X_test_normalized)

# Convert probabilities to class predictions
y_pred_cnn = np.argmax(y_pred_cnn_probs, axis=1)

# Decode numerical predictions back to original labels
y_pred_cnn_labels = label_encoder.inverse_transform(y_pred_cnn)

# Print classification report for CNN
print("Classification Report (CNN):")
print(classification_report(y_test, y_pred_cnn_labels, target_names=np.unique(y)))

# Print confusion matrix for CNN
conf_matrix_cnn = confusion_matrix(y_test, y_pred_cnn_labels, labels=np.unique(y))
print("Confusion Matrix (CNN):")
print(conf_matrix_cnn)

# Calculate and print accuracy, precision, recall, and F1-score for CNN
accuracy_cnn = accuracy_score(y_test, y_pred_cnn_labels)
precision_cnn = precision_score(y_test, y_pred_cnn_labels, average='weighted')
recall_cnn = recall_score(y_test, y_pred_cnn_labels, average='weighted')
f1_cnn = f1_score(y_test, y_pred_cnn_labels, average='weighted')

print(f"Accuracy (CNN): {accuracy_cnn:.2f}")
print(f"Precision (CNN): {precision_cnn:.2f}")
print(f"Recall (CNN): {recall_cnn:.2f}")
print(f"F1 Score (CNN): {f1_cnn:.2f}")
