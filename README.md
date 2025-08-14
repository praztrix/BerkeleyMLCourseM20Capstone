### Evaluation of Image Classification Models

**Author: Prasanna Iyengar**

### Executive summary

The ability to accurately understand image content holds immense value across diverse applications. Despite the availability of numerous open-source image classification architectures, their practical performance—particularly concerning memory footprint, CPU utilization, inference latency, and classification accuracy—varies significantly. This project addresses this question by evaluating these key metrics for a selection of prominent Convolutional Neural Networks (CNNs) utilizing transfer learning with a custom dataset. The analysis will provide insights into the trade-offs between different model architectures, enabling informed decision-making for deploying  image classification solutions.

### Rationale
Why should anyone care about this question?


Understanding the content and themes within images offers significant value across various applications. While numerous open-source image classification architectures exist, their performance varies considerably in terms of memory footprint, latency, and accuracy. This project aims to systematically evaluate these critical metrics for four distinct model architectures, providing insights into their practical suitability for image classification tasks.

 
### Research Question

What are you trying to answer?


#### Evaluate Key Metrics for Image Classification Models with Transfer Learning

Analyze the performance of the following models using transfer learning for image classification, focusing on key metrics such as accuracy, precision, recall, inference latency, and memory footprint:

1. Base Model (Custom-built reference model)

2. ResNet50V2

3. DenseNet121

4. MobileNetV2

5. EfficientNetB0


#### Understand and implement various aspects of the Keras library for creating and manipulating Convolutional Neural Networks (CNNs) in the context of image classification. This includes:

1. Transfer Learning: Applying pre-trained models to new datasets.

2. Image Augmentation: Techniques to expand training data and improve model generalization.

3. Hyperparameter Tuning: Optimizing CNN performance through systematic parameter selection.

4. Model Performance and Accuracy Validation: Rigorously assessing model effectiveness using a custom dataseti



### Data Sources

#### CIFAR-10: Training and Initial Testing Data
The CIFAR-10 dataset is a widely recognized benchmark in computer vision for image classification. It consists of 60,000 32x32 pixel color images categorized into 10 distinct classes. Each class contains 6,000 images, equally split into 50,000 images for the training set and 10,000 images for the test set.

The 10 classes are:

airplane

automobile

bird

cat

deer

dog

frog

horse

ship

truck

For this project, CIFAR-10 serves as the primary dataset for training the base model and the transfer-learned architectures (ResNet50V2, DenseNet121, MobileNetV2, EfficientNetB0) and for their in-training validation.

#### CINIC-10: Independent Performance Evaluation Data

[CINIC-10](https://www.kaggle.com/datasets/mengcius/cinic10) an extensive dataset designed as a "bridge" between CIFAR-10 and the much larger ImageNet dataset. It comprises a total of 270,000 images, which is 4.5 times larger than CIFAR-10. These images are also 32x32 pixels color images and belong to the same 10 classes as CIFAR-10. The dataset is constructed by combining all images from CIFAR-10 with a selection of downsampled images from the ImageNet database.

CINIC-10 is equally split into three subsets: training, validation, and test, each containing 90,000 images. Within each subset, there are 9,000 images per class.

For this project, **500 images of CINIC-10 test set of every class**  will be used to evaluate the final performance of the models trained on CIFAR-10. This allows us to assess how well the models generalize to a broader and potentially more diverse set of unseen images, which is critical for real-world application where data distribution might subtly shift. This setup helps to expose potential overfitting to the CIFAR-10 training data and provides a more robust measure of the models' true generalization capabilities.

#### Exploratory Data Analysis 

The following is observed for the CIFAR-10 and the subset of CINIC-10 dataset used in this evaluation.

**Data Loading and Initial Inspection**

Loading:  CIFAR-10 database consists of 50,000 training images and 10,000 test images, each 32×32 pixels in size with three color channels (RGB).

Shape: (50000, 32, 32, 3) for the training images indicates a dataset of 50,000 images, each 32×32 pixels with 3 color channels.  (10000, 32, 32, 3) for the test images indicates a dataset of 10,000 images, each 32×32 pixels with 3 color channels.

Loading: 5000 test images for performance evaluation after training of shape (32, 32, 3)

**Class Distribution Analysis**

The CIFAR-10 dataset exhibits a perfectly balanced class distribution. Each of the 10 classes contains an equal number of samples: 5,000 images per class in the training set and 1,000 images per class in the test set. This balance simplifies initial model training as it prevents bias towards any particular class due to unequal representation.

**Pixel Intensity Distribution**

Analysis of pixel intensity provides insight into the raw value range and distribution across color channels. Histograms are generated for the Red, Green, and Blue color channels by flattening the image arrays and plotting the frequency of pixel intensity values (ranging from 0 to 255). For CIFAR-10, we typically observe a relatively even distribution of pixel values across all three channels, indicating a good spread of color information without heavy skewing towards very dark or very bright images.

When the pixel intensity distribution is not uniform across an image dataset, it can impact the training covergence time and the performance of Convolutional Neural Networks (CNNs). 

Ref: [Pixel Intensity Histogram Characteristics: Basics of Image Processing and Machine Vision](https://www.allaboutcircuits.com/technical-articles/image-histogram-characteristics-machine-learning-image-processing/) 

**Sample Image Visualization**

 This is a sanity check to ensure the images are loaded correctly and the labels match the images. It also gives you a visual feel for the type and quality of the data your model will be trained on.


### Methodology

#### Data Loading and Preprocessing (load_and_preprocess_dataset, load_cinic10_test_dataset, preprocess_func_base and others)

This initial stage handles the acquisition and preparation of image data for model consumption.

**CIFAR-10 Loading and Augmentation**: The load_and_preprocess_dataset function loads the CIFAR-10 dataset. It then performs crucial preprocessing steps:

**One-Hot Encoding**: Converts integer labels (e.g., 0, 1, 2) into a binary vector format (e.g., [1, 0, 0], [0, 1, 0]), which is suitable for categorical cross-entropy loss.

**Image Resizing**: All images are resized to a TARGET_SIZE (e.g., 224x224 pixels), ensuring uniform input dimensions for the models. ResNet, MobileNet, DenseNet and EfficientNetB0 operate on imagenet data of the dimension 224x224.

**Model-Specific Preprocessing**: A preprocess_func (like preprocess_func_base or those for pre-trained models, e.g., tf.keras.applications.resnet50.preprocess_input) is applied. This typically involves pixel scaling (e.g., normalizing to [0, 1] or [-1, 1]) and other specific operations required by the pre-trained models.

**Data Augmentation**: For the training dataset, a keras.Sequential model applies random transformations such as horizontal flips, rotations, and zooms. This artificially expands the dataset, helping the model generalize better and reducing overfitting.

**Dataset Pipelining**: tf.data.Dataset is used to create efficient data pipelines, including shuffling (for randomness during training), batching (grouping samples for parallel processing), and prefetching (overlapping data loading and model execution). When tf.data.Dataset is used, this batch size overrides the batch size used in the fir function.

**CINIC-10 Test Data Loading**: The load_cinic10_test_dataset function specifically loads the CINIC-10 test set. This dataset is kept separate from the training data and is used for an independent, final evaluation of the trained models to assess their generalization capabilities on a  subset of larger distribution of images within the same classes. It undergoes resizing and model-specific preprocessing but not augmentation, as it's for evaluation.



#### Model Pipeline Execution (run_model_pipeline Function)
The run_model_pipeline function orchestrates the entire process for each specified model (BaseModel, ResNet50V2, DenseNet121, MobileNetV2, EfficientNetB0).


**Hyperparameter Tuning (Keras Tuner)**:

Use keras_tuner.RandomSearch to find the best combination of hyperparameters (e.g., learning rate, number of units in dense layers) based on val_accuracy on the CIFAR-10 validation set. CIFAR-10 test set is used as a validation set. The following two hyper parameters are considered:

1. learning rate
2. number of units in dense layers


**Search Process**: The tuner searches across defined ranges for hyperparameters, training multiple models for a few epochs to identify promising combinations.

**Model Selection and Compilation**: After the search, the best performing model and its hyperparameters are retrieved from the tuner. The model is then compiled with the optimal learning rate, a categorical cross-entropy loss function, and accuracy metrics. Optionally, best performing models are written to incurring disk for later analysis without training time.

**Final Model Training**: The selected best model is trained for FINAL_EPOCHS on the preprocessed and augmented CIFAR-10 training dataset, with validation on the CIFAR-10 test set. Training time is recorded.

**Visualization**: A bar plot visualizes the validation accuracy of different hyperparameter trials, providing an immediate overview of the tuning results. After training, plots are generated to visualize the training and validation loss and accuracy curves over epochs. This helps in understanding learning progress and identifying overfitting or underfitting.

For transfer learning models (ResNet50, MobileNetV2, EfficientNetB0, DenseNet121), the pre-trained convolutional base is loaded with imagenet weights and initially frozen (base_model.trainable = False), and only the newly added classification layers are tuned.

### Model Evaluation and Analysis
After training, the models undergo a comprehensive evaluation phase.

**CIFAR-10 Test Set Evaluation**: The trained model's performance is first evaluated on the CIFAR-10 test dataset to get final loss and accuracy metrics.

**CINIC-10 Test Set Evaluation**: The model is then evaluated on the CINIC-10 test dataset. This separate evaluation provides an assessment of the model's generalization ability to unseen data that might have different underlying characteristics than the CIFAR-10 training set.

**Key Metrics for Evaluation**

1. Total Parameters
2. Training Time (s)
3. Tuner Search Time (s)
4. Train Accuracy
5. Train Loss
6. Validation Accuracy
7. Validation Loss
8. CINIC-10 Test Accuracy
9. CINIC-10 Test Loss
10. Inference Time Per Sample (s)
11. In-Memory Model Size (MB)
12. File-Based Model Size (MB)
13. Confusion Matrix and Classification Report





### Results

![Model Performance](https://github.com/praztrix/BerkeleyMLCourseM20Capstone/images/CIFAR10MultiModelPerformancePlots.png "Model Performance")

#### Performance Analysis of Models

- Refer to [CIFAR-10MultiModelTransferLearningEvaluation.ipynb](https://github.com/praztrix/BerkeleyMLCourseM20Capstone/CIFAR-10MultiModelTransferLearningEvaluation.ipynb) for performance analysis of models. 

#### Overall Analysis

- The accuracy of the base model is low and therefore is not analyzed for other metrics.
- The validation accuracy of ResNet50, EfficientNetB0, and DenseNet121 models is comparable.
- ResNet50 has the highest training accuracy.
- The CINIC10 accuracy of ResNet50, EfficientNetB0, and DenseNet121 models is comparable.
- The ResNet50 model has the lowest training loss.
- The MobileNetV2 model has the lowest inference time per sample. This is not surprising as it was developed for mobile devices with resource constraints.
- The ResNet50 model consumes about three times more memory than the next highest memory-consuming model. This is due to the depth of the ResNet50 architecture.  

#### Model Selction
- Based on accuracy and loss, ResNet50  and EfficientNetB0 models are comparable.
- Inference time per image is slightly better for ResNet50, whereas the EfficientNetB0 model is better at memory and disk consumption.
- CINIC10 performance on EfficientNetB0 was slightly better than ResNet50, particularly in terms of validation loss.
-  The EfficientNetB0 model will also work better on mobile devices due to its low memory footprint.
- **EfficientNetB0** is the final choice as it can also be used on mobile devices.


### Next steps

- Run more trials for Keras search.
- Train the best-fit model for more epochs (30+). Ten epochs were chosen due to compute considerations. Early Stopping could be used to make efficient use of comoute resources.
- Train with a larger subset of data from the CINIC10 dataset and use CIFAR10 for performance validation. This is a function of compute and memory availability.

### Outline of project

Repository: 

[BerkeleyMLCourseM20Capstone](https://github.com/praztrix/BerkeleyMLCourseM20Capstone) 

Files:

- [capstone_utils.py](https://github.com/praztrix/BerkeleyMLCourseM20Capstone/blob/main/capstone_utils.py) - Utility functions
- [custom_cinic10_data.zip](https://github.com/praztrix/BerkeleyMLCourseM20Capstone/blob/main/custom_cinic10_data.zip) - Contains a subset of CINIC10 files for the performanve evaluation of models.
- [EDAForCIFAR-10MultiModelTransferLearningEvaluation.ipynb](https://github.com/praztrix/BerkeleyMLCourseM20Capstone/bolb/main/EDAForCIFAR-10MultiModelTransferLearningEvaluation.ipynb) - Notebook for EDA
- [CIFAR-10MultiModelTransferLearningEvaluation.ipynb](https://github.com/praztrix/BerkeleyMLCourseM20Capstone/blob/main/CIFAR-10MultiModelTransferLearningEvaluation.ipynb) - Notebook for model evaluation and performance analysis.

**All my Notebooks were run on Google Colab with Colab Pro subscription. The CPU type was A-100.**
Even with an A-100 CPU type on Google Colab, the total training time for 5 models took about 27 mins. The EDA also took some time as the pixel intensity analysis is compute intensive.

#### Instructions to Run these Notebooks on Google Colab
1. `capstone_utils.py`  should be copied to the `/content` folder on Google Colab run time.
2. `custom_cinic10_data.zip` should be copied to the `/content` folder on Google Colab run time. The Notebook will unzip this file for performance evaluation.
3. `capstone_utils.py` and `custom_cinic10_data.zip` should be copied to the `/content` folder on Google Colab after a new runtime is started. 


**Note**: Colab deletes all files uploaded to it when a runtime is deleted or the CPU/GPU is disconnected. Re-upload of `capstone_utils.py` and `custom_cinic10_data.zip` (along with unzipping operation) is required in that case.
