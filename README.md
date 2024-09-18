
# Cat and Dog Breed Classification using CNN

This repository contains the implementation of a Convolutional Neural Network (CNN) model for classifying cat and dog breeds. The project is part of my college AI coursework, utilizing a dataset sourced from Kaggle.

## Dataset

The dataset used in this project is the [Animal Breed - Cats and Dogs](https://www.kaggle.com/datasets/imsparsh/animal-breed-cats-and-dogs) dataset from Kaggle. It contains images of different cat and dog breeds, organized into respective directories for training and testing.

## Project Overview

The main goal of this project is to build a CNN model that can accurately classify cat and dog breeds from images. This involves:

- Preprocessing image data
- Building and training a CNN model
- Evaluating model performance
- Improving classification accuracy through tuning

### Features

- **CNN Architecture**: Built using TensorFlow and Keras for efficient feature extraction and classification.
- **Google Colab**: Used for model training and experimentation via Jupyter notebooks.
- **Visual Studio Code**: Used for development and debugging when needed.
- **Anaconda**: Managed dependencies and Python environment.
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score.

## Installation

To set up your environment and run this project, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/alharitz/Ai-Project-Cat-and-Dog-Classification.git
   ```
2. **Navigate to the Project Directory**:
   ```bash
   cd Ai-Project-Cat-and-Dog-Classification
   ```
3. **Set Up the Environment**:
   - If you are using Anaconda, create and activate a new environment:
     ```bash
     conda create --name cnn-env python=3.8
     conda activate cnn-env
     ```
   - Install the required dependencies using `requirements.txt`:
     ```bash
     pip install -r requirements.txt
     ```

4. **Download the Dataset**:
   - Obtain the dataset from [Kaggle](https://www.kaggle.com/datasets/imsparsh/animal-breed-cats-and-dogs) and place it in the appropriate folder.

5. **Open the Jupyter Notebook**:
   - You can use Google Colab by uploading the notebook and dataset files to your environment.
   - Alternatively, if running locally, launch the notebook using:
     ```bash
     jupyter notebook
     ```

## Usage

1. Run the entire notebook `cat_dog_classification.ipynb` to preprocess the data, train the model, and evaluate the results.
2. Modify any hyperparameters, such as batch size or learning rate, in the notebook if necessary.
3. After training, the model evaluation and classification results will be displayed in the notebook.

## Model Architecture

The CNN model consists of:

- Multiple convolutional layers for feature extraction
- Max pooling layers to reduce dimensionality
- Dense layers for classification
- Softmax output layer for multi-class classification

## Results

The model achieved promising classification accuracy on the test dataset. Further fine-tuning and hyperparameter optimization can improve the performance.

### Example Output:
```
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 52ms/step
302.jpg: birman
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 51ms/step
529.jpg: siamese
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 40ms/step
866.jpg: bengal
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 48ms/step
253.jpg: samoyed
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 41ms/step
642.jpg: siamese
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 42ms/step
11.jpg: samoyed
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 45ms/step
1022.jpg: keeshond
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 44ms/step
```

## Challenges

- **Class Imbalance**: Handled using data augmentation and class weighting.
- **Model Optimization**: Fine-tuned model hyperparameters to enhance performance.

## Contributing

If you'd like to contribute, please fork the repository and submit a pull request. You can also raise issues for any bugs or feature requests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

**Author**: [Your Name](https://www.linkedin.com/in/your-linkedin)
```

This updated `README.md` includes a section on setting up the environment with `requirements.txt`. Adjust the details as needed for your specific implementation.
