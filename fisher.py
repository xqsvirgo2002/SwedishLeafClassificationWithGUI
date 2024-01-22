from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.preprocessing import label_binarize
from skimage import io, color, exposure, transform
import time
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

def load_images(folder):
    start_time_for_load_image = time.time()
    images = []
    labels = []
    class_names = sorted(os.listdir(folder))
    for i, label in enumerate(class_names):
        label_path = os.path.join(folder, label)
        if os.path.isdir(label_path):
            for filename in os.listdir(label_path):
                img_path = os.path.join(label_path, filename)
                img = io.imread(img_path)
                img = preprocess_image(img)
                images.append(img)
                labels.append(i)
    end_time_for_load_image = time.time()
    print(f'image loading Time: {end_time_for_load_image - start_time_for_load_image:.2f} seconds')
    return np.array(images), np.array(labels), class_names

def preprocess_image(img):
    img = transform.resize(img, (64, 64))
    img = color.rgb2gray(img)
    img = exposure.equalize_adapthist(img)
    return img.flatten()

def print_accuracy(title, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    print(f'{title} Accuracy: {acc:.2f}')
def plot_roc_curve(y_true, y_scores, title, class_names):
    fpr, tpr, _ = roc_curve(y_true.ravel(), y_scores.ravel())
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, lw=2, label=f'{title} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {title}')
    plt.legend(loc='lower right')
    plt.show()

data_folder = "/Users/xiaoqisheng/Downloads/Swedish_leaf_dataset"
images, labels, class_names = load_images(data_folder)
# 分训练和测试集
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# ROC curve
y_train_bin = label_binarize(y_train, classes=np.unique(y_train))
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))

# Fisher Linear Discriminant
start_time = time.time()
fisher_lda = LDA()  # 使用 LSQR 求解器和自动缩减
fisher_lda.fit(X_train, y_train)
y_scores_lda = fisher_lda.decision_function(X_test)
y_pred_lda = fisher_lda.predict(X_test)
end_time = time.time()

print(f'Fisher LDA Time: {end_time - start_time:.2f} seconds')
plot_roc_curve(y_test_bin, y_scores_lda, 'Fisher LDA', class_names)
print_accuracy('Fisher LDA', y_test, y_pred_lda)
