# Age_estimation_using_CNN
Automatic age estimation of facial images is an important problem in computer vision and image analysis. It has a large number of potential application areas. To estimate near accurate age from facial images require a large amount of data and tedious training period. In this project, an age estimator is proposed based on the convolutional neural network that can predict age from facial images almost accurately. This approach requires less training data than the related works yet keeping a low mean absolute error.

A model is built on top of ResNet50 that implements age estimation as a regression problem. The experimental results are compared with other age estimation techniques. The comparison exhibits that the performance of the age estimation system is close to other related works even after training with a relatively smaller dataset.
# Keywords: 
Age estimation, Convolutional Neural Network, Face Detection, ResNet50.
# Dataset
UTKFace dataset for training, appa real for validation and FG-NET for testing.
# Used libraries: 
pandas, numpy, cv2, dlib, keras, Augmentor, ResNet50
# Preprocessing
The preprocessing phase consists of face detection, resizing the images and one hot encoding.
# Training
The CNN model consists of a ResNet50 model and a densely connected layer. The final layer is removed, and a new dense layer is added. The weights are initialized using ImageNet. The network consists of 50 layers, with images filtered in convolutional layers and ReLu as the activation function. The output is passed to the appended dense layer, with softmax activation applied to each node. The network's loss is calculated using the predicted and true values, and the network is optimized by adjusting weights in each iteration. Both Adam and SGD have been tried as optimizers. After iterating with all inputs in the training set, an epoch is completed, and the network predicts the age of images absent during the epoch. The model is saved if the MAE is better than previous epochs.
# Testing
After completing 20 epochs, the MAE of the model is calculated using testing dataset. During testing, the best model is saved in the training phase. The model predicts the ages of all input face images from the test set. Using the predicted ages and actual ages given in the dataset, MAE and CS have been calculated.
# Final Output
4.49 MAE and 67.37% cumulative score is acgieved using SGD optimizer which is better than the MAE and cs achieved by using adam optimizer.
