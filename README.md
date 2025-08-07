# Weather Image Classification with Transfer Learning

This repository presents a deep learning project that classifies weather conditions—Cloudy, Rain, Shine, Sunrise—from real-world images using transfer learning models. https://www.kaggle.com/datasets/pratik2901/multiclass-weather-dataset

## OVERVIEW

This project investigates whether modern computer vision models can accurately distinguish different weather conditions in natural images. Images were organized into four classes (Cloudy, Rain, Shine, Sunrise) and used to train and compare three pre-trained transfer learning models: MobileNetV2, ResNet50, and DenseNet121.
Among the three, MobileNetV2 consistently achieved the highest classification performance, with near-perfect AUC scores across all classes. DenseNet121 also performed well but lagged slightly behind. ResNet50 struggled more with class separation. These results suggest that lightweight models like MobileNetV2 are highly effective in visual classification tasks involving natural environmental conditions.

## SUMMARY OF WORK DONE

### Data

  * **Type:**
    * Input: A directory of natural scene images categorized into four weather classes: Cloudy, Rain, Shine, and Sunrise
    * Output: Predicted weather condition label, associated confidence score, and correctness indicator (correct/incorrect)
  * **Source:**
    * Custom image dataset organized into subfolders by class, collected from a weather image dataset (with images manually inspected and cleaned)
  * **Size of Classes:**
    * Cloudy: 271 images
    * Rain: 174 images
    * Shine: 219 images
    * Sunrise: 270 images
    * Total: 934 images
  * **Splits:**
    * Training: 80%
    * Validation: 20%
   
#### Compiling Data and Image Pre-processing

* **Data Collection:**
    * Dataset Source: A pre-organized weather image dataset consisting of four weather conditions: Cloudy, Rain, Shine, and Sunrise
    * Organization: Images were grouped into corresponding subdirectories for automatic label assignment using image_dataset_from_directory()
    * Balance: Classes were moderately balanced, with a slight variation in total image count per label
* **Data Cleaning:**
    * Manual Review: Images were inspected for quality, and corrupt or irrelevant examples were removed
    * Issues Addressed:
      * Removed grayscale or unreadable files that caused channel mismatch errors during model loading
      * Ensured consistent RGB format across the dataset to work with pretrained models
    * Fix Applied: A custom preprocessing function was used to convert any grayscale images into 3-channel RGB format
* **Image Pre-processing:**
    * Resizing: All images were resized to 180×180 pixels to standardize input dimensions
    * Normalization: Pixel values were scaled to the range [0, 1] by applying Rescaling(1./255)
    * Augmentation (Training Only):
      * RandomFlip("horizontal"): To simulate variation in orientation
      * RandomRotation(0.1): To add rotational variance
      * RandomZoom(0.1): To account for distance variations
    * Batching and Shuffling:
      * Images were loaded in batches of 32
      * Data was shuffled to reduce overfitting and improve generalization

#### Data Visualization

<img width="465" height="199" alt="Screenshot 2025-08-04 at 5 28 08 PM" src="https://github.com/user-attachments/assets/398a72d9-15e2-4f98-bfc2-c29f6d0c8e93" />

  <img width="834" height="815" alt="Screenshot 2025-08-04 at 5 03 29 PM" src="https://github.com/user-attachments/assets/00b447f5-8141-4f7e-b1de-985c8cc73f9c" />

### Problem Formulation
The goal of this project is to classify images into four distinct weather conditions: Cloudy, Rain, Shine, and Sunrise. This is formulated as a multi-class image classification task, where the model receives an RGB image as input and outputs a predicted weather category.

* **Models Used:**
  To evaluate performance across different architectures, we trained and compared three models:
  * **Custom Convolutional Neural Network (Base)**
   * A small CNN built from scratch using Conv2D, MaxPooling2D, and Dense layers. It served as our performance benchmark.
  * **MobileNetV2 (Transfer Learning)**
   * A lightweight, efficient deep learning model pretrained on ImageNet. Fine-tuned for our dataset and showed the strongest performance across most classes.
  * **ResNet50 (Transfer Learning)**
   * A deeper residual network with skip connections. While powerful, it underperformed relative to the others on this task.
* **Loss Function & Optimizer:** SparseCategoricalCrossentropy(from_logits=True) and Adam optimizer with learning rate 0.001
* **Epochs:** Up to 20 with EarlyStopping 
* **Callbacks:** EarlyStopping: with patience=5, to prevent overfitting, ModelCheckpoint: to save the best performing model on the validation set

### Training

Model training was conducted on a MacBook Pro using an M2 chip and 16 GB of RAM, utilizing Jupyter Notebook and Google Colab as needed for GPU support. The training environment included essential libraries such as TensorFlow/Keras, NumPy, Pandas, Matplotlib, and Seaborn.

Each model was trained using the Adam optimizer (learning rate = 0.001) and SparseCategoricalCrossentropy as the loss function. Training and validation performance were tracked using accuracy and loss curves, with EarlyStopping and ModelCheckpoint callbacks to avoid overfitting and preserve the best-performing models.

Training time varied by model:
The Custom CNN trained quickly (~4 minutes).
MobileNetV2 completed in ~6 minutes.
ResNet50 took approximately 12–15 minutes due to its depth and complexity.

Each model was trained for up to 20 epochs, with patience set to 5, allowing early termination if validation loss failed to improve. The models were trained on weather images resized to 224×224 pixels, with batch sizes of 10 or 32, depending on the setup.

**Challenges:** A notable challenge was the visual similarity between certain classes—especially between “Shine” and “Sunrise”, both of which include clear skies and bright lighting. This caused misclassification and lower recall for the "Shine" class.
Another challenge was ensuring balance across classes, especially since the number of images varied slightly between them. Data augmentation helped address this by artificially increasing variety in training images.

### Performance Comparison

<img width="1135" height="462" alt="Screenshot 2025-08-04 at 5 26 12 PM" src="https://github.com/user-attachments/assets/0443d1ae-903a-42a9-9267-40ea03b3956a" />

<img width="1128" height="463" alt="Screenshot 2025-08-04 at 5 26 41 PM" src="https://github.com/user-attachments/assets/98f505f0-71ca-419d-8d59-ad0a3fd86ba2" />

<img width="1122" height="461" alt="Screenshot 2025-08-04 at 5 27 08 PM" src="https://github.com/user-attachments/assets/b3ff0b2f-012b-408d-b9a4-604bdcb20133" />

<img width="1119" height="450" alt="Screenshot 2025-08-04 at 5 30 55 PM" src="https://github.com/user-attachments/assets/fefd774f-0d79-4ee2-990e-2659aa926087" />

<img width="1126" height="455" alt="Screenshot 2025-08-04 at 5 31 17 PM" src="https://github.com/user-attachments/assets/848d97c3-c24a-4f00-b4f5-11b159edd21b" />

<img width="1031" height="599" alt="Screenshot 2025-08-04 at 4 12 44 PM" src="https://github.com/user-attachments/assets/9248c9c3-7b3c-4acf-9dcb-186677895c81" />

This plot is a ROC (Receiver Operating Characteristic) curve comparison of three transfer learning models — MobileNetV2, ResNet50, and DenseNet121 — on a multi-class image classification task (e.g., predicting weather classes like Cloudy, Rain, Shine, Sunrise).


### Conclusions

From the experiments conducted, MobileNetV2 emerged as the most effective model, achieving the highest overall performance in terms of AUC scores, F1-score, and validation consistency across all four weather classes: Cloudy, Rain, Shine, and Sunrise. It demonstrated robust generalization capabilities, particularly excelling in distinguishing "Rain" and "Sunrise" with AUC values near 0.99.

While DenseNet121 also performed well—especially in accurately classifying "Cloudy" and "Sunrise"—it slightly lagged behind MobileNetV2 on average. ResNet50, in contrast, showed the weakest performance, with AUC scores dropping as low as 0.59 for some classes, and displayed higher variance in predictions.

The custom CNN baseline model, though lightweight and fast to train, underperformed compared to the transfer learning models, highlighting the value of leveraging pretrained feature extractors for image classification.

Despite the overall success of MobileNetV2, the model occasionally struggled to distinguish between “Shine” and “Sunrise”, which share similar lighting and visual characteristics. This suggests that while transfer learning is powerful, further improvements could be achieved by: incorporating more diverse training data, applying advanced augmentation strategies, and experimenting with attention mechanisms or domain-specific preprocessing

Overall, this study demonstrates the feasibility of using deep learning models for weather classification from images, with MobileNetV2 serving as a strong baseline for future work.


### Future Work
 
* **Expand the Dataset:** Increase the volume of weather images per class to better support more complex models and reduce overfitting, particularly for underrepresented classes like Shine.
* **Broaden Visual Diversity:** Collect images from a wider range of geographic and seasonal conditions to introduce greater variation within each weather class (e.g., different types of cloud coverage or rainfall intensity).
* **Experiment with Attention-Based Architectures** Explore models like Vision Transformers (ViTs) or attention-based CNNs to better focus on localized visual cues (e.g., sunlight rays or raindrops).


## HOW TO REPRODUCE RESULTS

### Overview of Files in Repository

The following files and notebooks document the complete development process of this weather classification project, from data loading to model evaluation:

* **TrainBaseModel.ipynb:** TTrains a custom Convolutional Neural Network (CNN) as a baseline model.
* **TrainBaseModelAugmentation.ipynb :** Trains the same CNN with image augmentation to evaluate its impact on performance.
* **Train-MobileNetV2.ipynb:** Fine-tunes a MobileNetV2 model using transfer learning.
* **Train-ResNet50.ipynb:** Implements and trains a ResNet50 model using transfer learning.
* **Train-DenseNet121.ipynb:** Trains DenseNet121 as a third transfer learning model.
* **CompareAugmentation.ipynb:** Loads and compares the baseline model with and without augmentation using ROC curves.

### Software Setup

This project was developed and executed in Jupyter Notebook. 

Core Libraries Used:
* tensorflow, keras
  * layers, models, optimizers
  * callbacks (EarlyStopping, ModelCheckpoint)
  * applications (MobileNetV2, ResNet50, DenseNet121)
  * preprocessing.image (load_img, img_to_array)
* numpy, pandas, matplotlib, os, random
* sklearn
  * metrics (classification_report, roc_curve, auc)
  * model_selection (train_test_split)

### Training

* All required packages were installed directly in the Jupyter Notebook environment using pip or pre-installed Colab packages.
* Data was prepared using the image_dataset_from_directory method, which automatically handled directory parsing and split the dataset into training and validation sets (80/20 split).
* Images were preprocessed by:
  * Resizing them to 180x180 pixels.
  * Normalizing pixel values to a [0, 1] range.
  * Augmenting training images with random flips, zoom, and rotations (in the augmentation notebook).
  * Batching with a size of 32 and shuffling during training.
* Models trained include:
  * A custom CNN baseline
  * MobileNetV2
  * ResNet50
  * DenseNet121
* All training was performed using TensorFlow/Keras, with:
  * EarlyStopping (patience = 3–5) to prevent overfitting.
  * ModelCheckpoint to save the best model based on validation accuracy.
  * SparseCategoricalCrossentropy as the loss function.
  * Adam optimizer with a learning rate of 0.001 or 0.0003 depending on the model.
* Training was conducted over multiple sessions using GPU-accelerated environments in Google Colab and/or local Jupyter environments with TensorFlow GPU support.


#### Performance Evaluation

* After training, each model’s performance was evaluated using the held-out validation set.
* Evaluation metrics included:
  * Accuracy and loss on the validation set
  * Precision, recall, and F1-score per class (via classification_report)
  * Confidence scores for each predicted class
* Scripts and notebook cells were created to:
  * Plot training and validation loss/accuracy curves for each model using matplotlib
  * Generate confusion matrices to visualize misclassifications
  * Plot ROC curves for each of the three models (MobileNetV2, ResNet50, DenseNet121) for deeper comparison
* These visualizations helped assess:
  * Class-wise strengths and weaknesses
  * Whether models were overfitting or underfitting
  * Which architecture generalized best

## CITATIONS
https://www.kaggle.com/datasets/pratik2901/multiclass-weather-dataset
