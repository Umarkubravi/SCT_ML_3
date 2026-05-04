# -----------------------------
# Improved SVM Cats vs Dogs (FULL REPLACEMENT)
# -----------------------------

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# -----------------------------
# SETTINGS
# -----------------------------
IMG_SIZE = 64
DATA_DIR = "dataset"
CATEGORIES = ["cats", "dogs"]

# -----------------------------
# LOAD DATA
# -----------------------------
data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DATA_DIR, category)
    label = CATEGORIES.index(category)

    for img in os.listdir(path):
        try:
            img_path = os.path.join(path, img)

            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

            data.append(image.flatten())
            labels.append(label)

        except:
            continue

data = np.array(data)
labels = np.array(labels)

print("Total Images Loaded:", len(data))

# -----------------------------
# SCALE DATA (IMPORTANT IMPROVEMENT)
# -----------------------------
scaler = StandardScaler()
data = scaler.fit_transform(data)

# -----------------------------
# SPLIT DATA
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# -----------------------------
# TRAIN MODEL (RBF KERNEL)
# -----------------------------
model = SVC(kernel='rbf')
model.fit(X_train, y_train)

# -----------------------------
# PREDICT
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# ACCURACY
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# -----------------------------
# SHOW RESULTS
# -----------------------------
for i in range(min(5, len(X_test))):
    img = X_test[i].reshape(IMG_SIZE, IMG_SIZE)

    plt.imshow(img, cmap="gray")
    plt.title("Predicted: " + CATEGORIES[y_pred[i]])
    plt.axis("off")
    plt.show()

input("Press Enter to exit...")