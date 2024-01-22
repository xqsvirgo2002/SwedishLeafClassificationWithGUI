import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.preprocessing import label_binarize
from skimage import io, color, exposure, transform
import time
from joblib import dump

def load_images(folder):
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
    return np.array(images), np.array(labels), class_names

def preprocess_image(img):
    img = transform.resize(img, (64, 64))
    img = color.rgb2gray(img)
    img = exposure.equalize_adapthist(img)
    return img.flatten()

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

def print_accuracy(title, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    print(f'{title} Accuracy: {acc:.2f}')

# 加载数据
data_folder = "/Users/xiaoqisheng/Downloads/Swedish_leaf_dataset"
images, labels, class_names = load_images(data_folder)

# 分训练和测试集
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
y_train_bin = label_binarize(y_train, classes=np.unique(y_train))
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))

# SVM模型
svm = SVC(probability=True, kernel='linear', C=374.6401188473625)
svm.fit(X_train, y_train)
y_scores_svm = svm.decision_function(X_test)
fpr_svm, tpr_svm, _ = roc_curve(y_test_bin.ravel(), y_scores_svm.ravel())
roc_auc_svm = auc(fpr_svm, tpr_svm)

# Fisher LDA模型
fisher_lda = LDA()
fisher_lda.fit(X_train, y_train)
y_scores_lda = fisher_lda.decision_function(X_test)
fpr_lda, tpr_lda, _ = roc_curve(y_test_bin.ravel(), y_scores_lda.ravel())
roc_auc_lda = auc(fpr_lda, tpr_lda)

dump(svm, 'svm_model.joblib')
dump(fisher_lda, 'lda_model.joblib')

# 计算准确率和权重
svm_accuracy = accuracy_score(y_test, svm.predict(X_test))
lda_accuracy = accuracy_score(y_test, fisher_lda.predict(X_test))
total_accuracy = svm_accuracy + lda_accuracy
svm_weight = svm_accuracy / total_accuracy
lda_weight = lda_accuracy / total_accuracy

# 计算组合概率
svm_proba = svm.predict_proba(X_test) * svm_weight
lda_proba = fisher_lda.predict_proba(X_test) * lda_weight
combined_proba = svm_proba + lda_proba

# 微观平均ROC曲线
fpr_dict = {}
tpr_dict = {}
num_classes = combined_proba.shape[1]  # 获取类别数量
for i in range(num_classes):
    fpr_dict[i], tpr_dict[i], _ = roc_curve(y_test_bin[:, i], combined_proba[:, i])

all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(num_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(num_classes):
    mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
mean_tpr /= num_classes
mean_auc = auc(all_fpr, mean_tpr)

# 绘制ROC曲线
plt.figure(figsize=(8, 8))
plt.plot(fpr_svm, tpr_svm, label=f'SVM (AUC = {roc_auc_svm:.2f})')
plt.plot(fpr_lda, tpr_lda, label=f'Fisher LDA (AUC = {roc_auc_lda:.2f})')
plt.plot(all_fpr, mean_tpr, color='blue', linestyle='-', linewidth=2, label='Combined Model (AUC = {0:0.2f})'.format(mean_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend(loc='lower right')
plt.show()