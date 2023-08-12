## Emotion Recognition using VGG16 and FER2013 Dataset

This repository contains code for training an emotion recognition model using the [FER2013](https://www.kaggle.com/datasets/msambare/fer2013) dataset and the [VGG16](https://arxiv.org/abs/1409.1556) architecture. The primary objective is to classify facial expressions into seven different emotions: angry, disgust, fear, happy, sad, surprise, and neutral.

### Hyperparameter Tuning

- *Learning Rate:* I set the learning rate of the Adam optimizer to 1e-4.
- *Batch Size:* I chose the batch size of 512 for both training and validation data generators.
- *Epochs:* The model was trained for 20 epochs.
- *Fine-tuning:* I set the first 10 layers until the layer "block4_conv1" as non-trainable.
- *Model complexity:* I chose the length of my hidden layer as 32, and FCL depth as 2.

### Model Architecture and Fine-Tuning
The core architecture of the model is based on the VGG16 architecture. The initial layers of VGG16 serve as feature extractors and are frozen. Remaining layers are trained along with fully connected layers, which consists of 32 node length layer with Leaky ReLu activation, a batch normalization layer, and a final 7 node (length of classes) length layer with softmax activation.

### Training
No data augmentation techniques were applied during training. The Adam optimizer was employed with a categorical cross-entropy loss function. Model was saved based on the best validation accuracy using the ModelCheckpoint callback.

### Results and Evaluation
After training for 20 epochs, the model demonstrated a significant overfitting with 0.99 accuracy on training set and 0.65 accuracy on validation set. No matter how I tuned the hyperparameters, I couldn't go above 0.65 accuracy on validation set or go below 1.100 on validation loss. Some of my decision was just poor, some of them I don't have enough compitational resources to implement them. 0.65 accuracy may sound "extremely low", however, human-level accuracy on FER2013 is only at 65±5% according to this [report](http://cs230.stanford.edu/projects_winter_2020/reports/32610274.pdf). So, my accuracy is just "low". Below are some example configurations and their accuracy after 20 epochs:

| Batch Normalization Layer | LR | Batch Size | Hidden Layer Length | Dropout Layer Rate | Regularization | Train Acc | Val Acc |
| ------------------------- | -- | ---------- | ------------------- | ------------------ | -------------- | --------- | ------- |
|-| 1e-4 | 512 | None | None | None | 0.25 | 0.25 |
|-| 1e-4 | 512 | 16 | None | None | 0.87 | 0.62 |
|-| 1e-4 | 512 | 16 | 0.3 | None | 0.83 | 0.62 |
|-| 1e-4 | 512 | 8 | None | None | 0.73 | 0.60 |
|-| 1e-3 | 512 | 8 | None | None | 0.25 | 0.25 |
|-| 1e-4 | 512 | 32 | None | None | 0.95 | 0.63 |
|-| 1e-4 | 512 | 32 | None | L2: 1e-2 | 0.91 | 0.61 |
|+| 1e-4 | 512 | 32 | None | None | 0.99 | 0.65 |

### Sample Predictions
Here are three sample images along with their corresponding model predictions:

| Image 1 | Image 2 | Image 3 |
| ------- | ------- | ------- |
| ![Image 1](images/jp.png) | ![Image 2](images/ww.png) | ![Image 3](images/gus.png) |

- Image 1 Prediction: Happy
- Image 2 Prediction: Sad
- Image 3 Prediction: Sad (Wrong)

### Usage
You can dowload the model from [here](https://drive.google.com/file/d/1PJ2Q8NgIRdRXREB37UyivO6Vsm9_7dH3/view?usp=sharing) and use it in like simple_predictions.ipynb. I didn't include the model in repo since it is more than 150MB.
