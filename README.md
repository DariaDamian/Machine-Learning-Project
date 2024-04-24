## Machine Learning Project: Image Classification of Clothing Articles

### Project Overview

This project explores advanced image classification techniques using three machine learning models to analyze a dataset of 15,000 labeled clothing images from the Zalando website. We implemented Feed-forward Neural Networks, Decision Trees, and Support Vector Machines to classify images into five clothing categories based on pixel values.

### Technologies Used

- **Python:** Main programming language for model implementation.
- **TensorFlow and Keras:** Utilized for building and training neural network models.
- **scikit-learn:** Used for implementing SVM and Decision Tree models.
- **PCA (Principal Component Analysis):** Employed for dimensionality reduction during preprocessing.

### Models and Methodology

1. **Neural Network:** Designed a Feed-forward neural network from scratch. The model includes an input layer, multiple hidden layers, and an output layer, using the Sigmoid activation function. Accuracy=84.34%.
2. **Support Vector Machine (SVM):** Implemented using the polynomial kernel to effectively handle the non-linear nature of image data. Accuracy=86.12%
3. **Decision Tree:** Built from scratch, focusing on simplicity and computational efficiency. Utilized Gini impurity for decision making at nodes.  Accuracy=76.82%

### Results

- The neural network and SVM models showed comparable accuracy, significantly outperforming the Decision Tree model.
- The best performing models were able to classify the clothing images with high accuracy, leveraging the specific characteristics of each algorithm.
  
### Future Work

- Experiment with convolutional neural networks which are better suited for image data.
- Explore alternative activation functions and more complex ensemble methods to improve model performance.
- Additional feature engineering and use of more sophisticated image preprocessing techniques.

### Repository Structure

```
/root
  ├── models
  │   ├── neural_network.py
  │   ├── support_vector_machine.py
  │   └── decision_tree.py
  ├── data
  │   └── fashion-mnist
  ├── notebooks
  │   └── model_evaluation.ipynb
  └── README.md
```

### How to Run

Details on environment setup, dependencies, and step-by-step instructions to train and evaluate the models are included in the project notebooks.

Notes:
For our image classification project, if your classes are balanced and misclassification costs are roughly equal, accuracy might suffice. 
