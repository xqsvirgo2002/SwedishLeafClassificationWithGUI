# SwedishLeafClassificationWithGUI

## Dataset
You can download the dataset from the following link: [Swedish Leaf Dataset](https://www.cvl.isy.liu.se/en/research/datasets/swedish-leaf/).

Before using the dataset, please make sure to follow these steps:

1. Download the dataset files.
2. Merge the contents of the `leaf1` to `leaf15` folders into a single folder named `Swedish_leaf_dataset`.
3. Place the `Swedish_leaf_dataset` folder in the same directory as your code.

The structure of the dataset folder should look like this:
- Swedish_leaf_dataset/
  - leaf1/
  - leaf2/
  - ...
  - leaf15/


Now you are ready to use the dataset for your project.

## Usage
- `gui.py`: This script can be directly executed and it will utilize the pre-trained models located in the same directory.

- `svm.py` and `lda.py`: These scripts are used for parameter tuning and training using the dataset.


- `model.py`: This script is used to train a voting model combining the results from SVM and LDA.


Make sure to replace `"svm_model.pkl"` and `"lda_model.pkl"` with the actual names of your saved SVM and LDA models.



