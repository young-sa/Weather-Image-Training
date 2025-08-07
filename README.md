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
