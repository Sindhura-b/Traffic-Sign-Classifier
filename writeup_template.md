
**Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Loading the data set
* Exploring, summarizing and visualizing the data set
* Designing, training and testing the model architecture
* Using the model to make predictions on new images
* Analyzing the softmax probabilities of the new images

[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

---

You're reading it! and here is a link to my [project code](https://github.com/Sindhura-b/Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. German traffic sign data is used in this project. It consists of 32x32 colored images belonging to 43 types of german traffic signs. The analysis of the dataset is done using numpy library rather than hardcoding manually. 

I calculated summary statistics of the traffic signs data set:

* The size of training set is 40469
* The size of the validation set is 
* The size of test set is 12630
* The shape of a traffic sign image is 32x32
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the training data set. It is a bar chart showing how the data ...

![alt text][image1]

###Design and Test a Model Architecture

####1. The code for this project is built based on the code provide for LeNet-Lab. Before definind the model architecture, the images are preprocessed using grayscale conversion and normalization techniques. As a first step, I decided to convert the images to grayscale because the classification of traffic signs is mostly dependent on the features and edges in an image and is independent of the color of the traffic sign. Hence, the unnecessary color information is avoided by converting the colored images to grayscale. In the second step, images are converted to grayscale as mormalization of the data makes it easier for the  optimizer to find a good solution. Normalization of the image dataset is done by subtracting and diving each pixel value by 128. 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


####2. My final model consisted of the following layers:

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
 

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| EPOCHS        		| 45   							| 
| Batch size       		| 128  							| 
| learning rate        		| 0.0097  							| 
| mean        		| 0   							| 
| standard deviation       		| 0.08  							| 

Before training the model, tensorflow variables 'x' and 'y' are set. Placeholder 'x' stores input batches and placeholder 'y' stores labels. Then, the labels are one-hot encoded using 'tf.one_hot' function, which performs binarization of categories. The ouput logits from model architecture are compared with the ground truth training labels to determine the cross entropy. The average cross entropy of all the images is the loss function which has to be minimized. This minimation is done using Adam optimizer that uses Adam algorithm. Adam optimizer is more sophisticated than stochastic gradient descent method and is a good defalut choice for optimization as suggested in the lectures. Finally, 'minimize' function is used to perform backpropagation and update the training loss. Then, evaluate function is built to evaluate the model prediction accuracy. Following this, model is trained and validated over the number of EPOCHS specified. Also, data is shuffled before the starting of this process to make it representative of the overall distribution of data and avoid repetation. 
To train the model, I used an ....

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.00
* validation set accuracy of 0.995 
* test set accuracy of 0.96

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


