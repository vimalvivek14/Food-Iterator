# Food-Iterator
A simple deep learning food identifier
Name of project:
Malaysian Food Detector

Description of the project:
Diagram:

Input and output information:
Input: Images of different classes of food
Output: Food label and nutritional info of that food taken from FatSecret database using the FatSecret API (*key and secret required)

Motivation for the idea:
To provide a fast and convenient way for users to check food calories by taking the photos of food eaten

Data Set Sources:
Kaggle Malaysia Food 11

Network Description:
VGG16 network use to classify images of food 
Model that won ILSVRC (Imagenet) competition in 2014

16 weight layers include convolution layers and fully-connected layers

Whole architecture of VGG16:
Conv3-64, conv3-64, maxpool, conv3-128, conv3-128, maxpool,  conv3-256, conv3-256, conv3-256, maxpool, conv3-512, conv3-512, conv3-512,maxpool,  conv3-512, conv3-512, conv3-512, maxpool, FC-4096, FC4096, FC-1000, soft-max.

Taken from model architecture (column D) from Table 1 of Very Deep Convolutional Networks for Large Scale Image Recognition, Simonyan and Zisserman (2014). Source: https://arxiv.org/abs/1409.1556

After fine-tuning, the output layer is replaced and the fc2 layer (second last layer) is unfrozen, so only the last two layers are trained in our training. Below is our model summary



Model Training:
3 epoch, around 6 hours

Testing:
Train accuracy: 0.9145
Test accuracy: 0.7903

Accuracy: 0.7958
Precision: 0.7997
Recall: 0.7958
F1 Score: 0.7951


Future Development:
Include more food in the database 
Develop a mobile application for users
Add tracking system to the calories consumed every day

List of Group members
Vimal, 
Aisyah Hafidzah Abdul Razak,
Elizabeth Ng Xin Yi
References if any:
www.fatsecret4j.com
https://github.com/CertifaiAI/TrainingLabs
