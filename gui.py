import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog
from PyQt5.QtGui import QPixmap
from skimage import io, color, exposure, transform
from joblib import load
import numpy as np
from PyQt5.QtCore import Qt

class LeafClassifierApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Leaf Classifier')
        self.setGeometry(300, 300, 350, 300)

        self.layout = QVBoxLayout()

        self.uploadButton = QPushButton('Upload Leaf Image', self)
        self.uploadButton.clicked.connect(self.uploadImage)

        self.imageLabel = QLabel(self)
        self.resultLabel = QLabel('Predicted Class: None', self)

        self.layout.addWidget(self.uploadButton)
        self.layout.addWidget(self.imageLabel)
        self.layout.addWidget(self.resultLabel)

        self.setLayout(self.layout)

    def uploadImage(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "All Files (*);;JPEG Files (*.jpeg);;PNG Files (*.png)", options=options)
        if fileName:
            self.displayImage(fileName)
            self.predictClass(fileName)

    def displayImage(self, path):
        pixmap = QPixmap(path)
        MAX_WIDTH = 300
        MAX_HEIGHT = 300
        pixmap = pixmap.scaled(MAX_WIDTH, MAX_HEIGHT, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        self.imageLabel.setPixmap(pixmap)
        self.imageLabel.setAlignment(Qt.AlignCenter)  # 设置居中对齐
        self.imageLabel.resize(self.width(), MAX_HEIGHT)  # 调整标签的大小

    def predictClass(self, image_path):
        img = io.imread(image_path)
        img_processed = self.preprocessImage(img)
        self.classifyLeaf(img_processed)

    def preprocessImage(self, img):
        img = transform.resize(img, (64, 64))
        img = color.rgb2gray(img)
        img = exposure.equalize_adapthist(img)
        return img.flatten()

    def classifyLeaf(self, image):
        # 加载模型（您需要先保存这些模型）
        svm = load('svm_model.joblib')
        lda = load('lda_model.joblib')

        # 示例权重和类别名称
        svm_accuracy = 0.93  # 示例值
        lda_accuracy = 0.8  # 示例值
        total_accuracy = svm_accuracy + lda_accuracy
        svm_weight = svm_accuracy / total_accuracy
        lda_weight = lda_accuracy / total_accuracy

        # 预测
        svm_pred = svm.predict_proba([image]) * svm_weight
        lda_pred = lda.predict_proba([image]) * lda_weight
        combined_pred = svm_pred + lda_pred

        # 示例类别名称
        class_names = [
            'Class1', 'Class2', 'Class3', 'Class4', 'Class5',
            'Class6', 'Class7', 'Class8', 'Class9', 'Class10',
            'Class11', 'Class12', 'Class13', 'Class14', 'Class15'
        ]
        predicted_class = class_names[np.argmax(combined_pred)]
        self.resultLabel.setText(f"Predicted Class: {predicted_class}")

def main():
    app = QApplication(sys.argv)
    ex = LeafClassifierApp()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
