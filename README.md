**Traffic Sign Recognition Project**

---

The goals / steps of this project are the following:
* Loading the data set
* Exploring, summarizing and visualizing the data set
* Designing, training and testing the model architecture
* Using the model to make predictions on new images
* Analyzing the softmax probabilities of the new images

[image1]: Data and classes.png "Training images along with their class ID's"
[image2]: ./data_visualization.png "Visualization"
[image3]: ./my_test_images.png "New test images"
[image4]: ./prediction_1.jpg "Softmax predictions for test image 1"
[image5]: ./prediction_2.jpg "Softmax predictions for test image 2"
[image6]: ./prediction_3.jpg "Softmax predictions for test image 3"
[image7]: ./prediction_4.jpg "Softmax predictions for test image 4"
[image8]: ./prediction_5.jpg "Softmax predictions for test image 5"

---

You're reading it! and here is a link to my [project code](https://github.com/Sindhura-b/Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

*Data Set Summary & Exploration*

1. German traffic sign data is used in this project. It consists of 32x32 colored images belonging to 43 types of german traffic signs. The analysis of the dataset is done using numpy library rather than hardcoding manually. Below are some random images picked from the training data set and their respective class ID's.

![alt text][image1]

I calculated summary statistics of the traffic signs data set:

* The size of training set is 27839
* The size of the validation set is 6960
* The size of test set is 12630
* The shape of a traffic sign image is 32x32
* The number of unique classes/labels in the data set is 43

2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the training, validation and test data sets. It is a bar chart showing how the data is distributed across various classes.

![alt text][./prediction_1.jpg]

*Designing and Testing a Model Architecture*

1. The code for this project is built based on the code provide for LeNet-Lab. Before definind the model architecture, the images are preprocessed using grayscale conversion and normalization techniques. As a first step, I decided to convert the images to grayscale because the classification of traffic signs is mostly dependent on the features and edges in an image and is independent of the color of the traffic sign. Hence, the unnecessary color information is avoided by converting the colored images to grayscale. In the second step, images are converted to grayscale as mormalization of the data makes it easier for the  optimizer to find a good solution. Normalization of the image dataset is done by subtracting and diving each pixel value by 128. 

2. My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 10x10x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x64 				|
| Flatten            |                   |
| Fully connected		| outputs 400        									|
| dropout            |  keep_probability=0.5 (training), keep_probability=0.1 (testing)   |
| Fully connected		| outputs 200        									|
| dropout            |  keep_probability=0.5 (training), keep_probability=0.1 (testing)   |
| Fully connected		| outputs 43        									|
| Softmax				|      									|
 
 
3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| EPOCHS        		| 45   							| 
| Batch size       		| 128  							| 
| learning rate        		| 0.0097  							| 
| mean        		| 0   							| 
| standard deviation       		| 0.08  							| 

Before training the model, tensorflow variables 'x' and 'y' are set. Placeholder 'x' stores input batches and placeholder 'y' stores labels. Then, the labels are one-hot encoded using 'tf.one_hot' function, which performs binarization of categories. The ouput logits from model architecture are compared with the ground truth training labels to determine the cross entropy. The average cross entropy of all the images is the loss function which has to be minimized. This minimation is done using Adam optimizer that uses Adam algorithm. Adam optimizer is more sophisticated than stochastic gradient descent method and is a good defalut choice for optimization as suggested in the lectures. Finally, 'minimize' function is used to perform backpropagation and update the training loss. Then, evaluate function is built to evaluate the model prediction accuracy. Following this, model is trained and validated over the number of EPOCHS specified. Also, data is shuffled before the starting of this process to make it representative of the overall distribution of data and avoid repetation. 

4. As mentioned previosuly, this project was built by initially taking Yan LeCun's model architecture for MNIST data classification as basis. By training this model architecture with the German traffic sign data, test set accuracy is obtained as 0.86. To acheive an accuracy above 0.93, i followed step-by-step approach for making changes to the model and monitering its prediction accuracy. At first, I preprocessed the raw data set through greyscale conversion and normalization. This improved the test accuracy to 0.88, which is not the target value. Then, I tuned hyperparameters like learning rate and variance to 0.0097 and 0.08. Reducing the learning rate makes the model to learn slowly and avoid overshooting the target and divergance. Decreasing variance ensures a more uncertain distribution of weights and allows the optimization to be more robust as the training progresses. Instead of further preprocessing the data through augmentation, I chose to modify the model architecture. I began this by adding dropout at the end of first two fully connected layers to eliminate over-fitting of the data. This brought a significant increase in the test set prediction accuracy to 0.93. Then, I got tempted to add dropout after each max-pooling layer to see further improvement in the performance. However, my test accuracy decresed this time. Adding another convolution and max-pooling layer improved the validation accuracy to 0.99 but the test accuracy reduced to 0.91 for 30 epochs due to over-fitting.

Experimenting this architecture with different number of epochs (upto 50, as I don't have access to GPU) and batch sizes, hasn't impacted the performance significantly. L2 regularization is also added to the cost function to penalize large errors. Finally, I decided to change the dimensions of each layer, which I started by increasing the depth of each convolution layer. The depth of first convolution layer was increased from 6 to 16, while the depth of second convolution layer was increased from 16 to 64. This led my model to perform much better, validation accuracy increased to 0.995 and test accuracy to 0.951 due to more number of parameters(weights and biases). I went forward by further increasing the depth of the convolutional layers. Once again, the accuracy hasn't improved, which could be due to overfitting. Also, this made the training process slower due to very high number of parameters for optimization. Hence, I chose to switch back to the previous depths and increased the number of epochs to 45. At last, I achieved test accuracy of 0.96, which is quite satisfactory.

My final model results were:
* training set accuracy of 1.00
* validation set accuracy of 0.995 
* test set accuracy of 0.96

*Testing the Model on New Images*

1. I chose nine different German traffic signs from web and tested them on my model to check the prediction accuracy. This data set includes images from following classes

Here are nine German traffic signs that I found on the web:

![alt text][image3] 

2. This new data set is also preprocessed before starting training and validation. The model was able to guess 9 of the 9 traffic signs, which gave a test accuracy of 1.0. The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This shows that the model performed very well on this new test set compared to the previous test set.

3. The top five soft-max predictions show that the model was able to predict the classes of the images in data set with 1.00 accuracy. The other four closest predictions are shown in code/html document.

Here are top five softmax predictions for 5 German traffic sign images:

![alt text][image4] ![alt text][image5] 

![alt text][image6] ![alt text][image7] 

![alt text][image8] 

