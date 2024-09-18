# Alzheimer's Disease Classification using Deep Learning

## Project Overview
This project explores the application of deep learning models for classifying Alzheimer's disease stages using brain imaging data. We compare the performance of multiple state-of-the-art convolutional neural network architectures across two datasets: a standard dataset and a pseudo-RGB augmented dataset.

## Datasets
1. **Standard Dataset:** Original brain imaging data without augmentation
2. **Pseudo-RGB Dataset:** Augmented version of the brain imaging data

The pseudo-RGB dataset preparation process is detailed in a linked repository: [Pseudo-RGB Dataset Preparation](https://github.com/masud1901/Alzheimer-Dataset-Preperation).

## Models Evaluated
We benchmarked several cutting-edge convolutional neural network architectures, including:
- VGG16
- ResNet50
- InceptionV3
- MobileNetV2
- Xception

## Methodology
1. **Data Preparation:**
   - Split the datasets into train, validation, and test sets (80%, 10%, 10%)
   - Applied data augmentation techniques including rotation, width/height shifts, shear, zoom, and horizontal flip

2. **Model Architecture:**
   - Used transfer learning with pre-trained weights from ImageNet
   - Customized the top layers for our specific classification task
   - Implemented fine-tuning by unfreezing and training the last few layers of the base model

3. **Training:**
   - Utilized categorical cross-entropy loss and various optimizers
   - Implemented callbacks for early stopping, model checkpointing, and learning rate reduction
   - Trained for a maximum of 200 epochs with a batch size of 32

4. **Evaluation:**
   - Computed comprehensive metrics including accuracy, precision, recall, F1-score, Cohen's Kappa, log loss, and Brier score
   - Generated confusion matrices and ROC curves for multi-class classification
   - Analyzed per-class performance and misclassification rates

## Repository Structure
For each model, the following files are generated:
- `[ModelName].ipynb`: Jupyter notebook containing all the code for data processing, model training, and evaluation
- `model_summary.txt`: Architecture summary of the model
- `[Optimizer]_metrics_[ModelName].csv`: Detailed metrics for the model trained with specific optimizer
- `best_optimizer_metrics_[Optimizer]_[ModelName].csv`: Best metrics achieved by the model with the best optimizer
- `hyperparameter.csv`: Hyperparameters used for model training
- `segmented_multiclass_metrics.csv`: Detailed metrics for all evaluated models
- `summary_metrics_[ModelName].csv`: Summary of metrics for the specific model
- `training_history.csv`: Epoch-wise training and validation metrics
- `multiclass_confusion_matrix.png`: Visualization of the confusion matrix
- `multiclass_roc_curves.png`: ROC curves for multi-class classification

## Usage
To replicate this study:
1. Clone this repository
2. Install the required dependencies (list them here or include a `requirements.txt` file)
3. Run the desired `[ModelName].ipynb` notebook
4. Explore the generated CSV files for detailed metrics and PNG files for visualizations

## Detailed File Descriptions
- `[Optimizer]_metrics_[ModelName].csv`: Contains epoch-by-epoch metrics for the model trained with a specific optimizer.
- `best_optimizer_metrics_[Optimizer]_[ModelName].csv`: Presents the best metrics achieved by the model using the optimal optimizer.
- `hyperparameter.csv`: Lists all hyperparameters used across different model trainings.
- `segmented_multiclass_metrics.csv`: Provides a comprehensive breakdown of metrics for each class and overall model performance.
- `summary_metrics_[ModelName].csv`: Offers a condensed view of key performance indicators for the specific model.
- `training_history.csv`: Records the training and validation metrics for each epoch during model training.

## Future Work
- Explore ensemble methods combining multiple model architectures
- Investigate the impact of different data augmentation techniques
- Extend the study to include additional neuroimaging modalities

## Acknowledgements
We acknowledge the contributions of various datasets, libraries, and tools used in this project. Special thanks to the creators of the pseudo-RGB dataset preparation process detailed in the linked repository.
