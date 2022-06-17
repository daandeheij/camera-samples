# Seminar Computer Vision Project Blog
*by group 47: Daan de Heij, Douwe Hoonhout (June 2022)*

##### Group Members Information
- Daan de Heij (4974301): D.w.j.j.deheij@student.tudelft.nl 
- Douwe Hoonhout (4393155): D.Hoonhout@student.tudelft.nl

<img src="https://i.imgur.com/9tPrpCx.gif" width=80px, style="margin-left:5px; margin-right:12px;margin-bottom:10px" />

This blog post will take you through the process of our project in finding and experimenting with lightweight models to correctly classify different types of garbage. We will start by explaining some different state-of-the art lightweight models used in mobiles. These models are called MobileNetV1 [1], MobileNetV2 [2] and MobileNetV3 [3]. These models will be compared with some heavier models to see what the impact is of having a lightweight model. In the experiment section the different models are compared in terms of computation cost and accuracy.

## Introduction
For our project we were interested in creating or finding a deep learning model that could classify different types of garbage. In the Netherlands, garbage should be seperated according to the types: Green-waste, Plastic Metal Drinkcartons  (PMD), Glass, Paper, Textile and Rest. However sometimes it is unclear which items belongs in which category. That is why it would be nice to have an mobile application that could make a picture and tell me in which category an item belongs. This asks for lightweight deep learning models that are efficient enough to run on mobile devices.

## Goal

To obtain a deep learning model that is able to classify a picture of a single item of waste into one of the following categories: `cardboard`, `glass`, `metal`, `paper` and `plastic`. Furthermore, the model should be sufficiently light-weight such that classifications can be performed on mobile devices. 

## Background information

### MobileNet
The first version of MobileNet was introduced in 2017. It follows the main concepts used in normal CNN but found a way to reduce the number of parameters without significant loss in performance. The main concept that is introduced in MobileNet is Depthwise Separable Convolutions. This type of convolution is a combination of pointwise and depthwise convolution and is known to be a efficient class of convolution. The depthwise convolution will preserve the number of channels whereas in normal convolution the number of output channels can grow really quickly. The pointwise convolution is then used to create the number of output channel that is required. Pointwise convolution uses a 1 x 1 kernel and is thus less computationally expensive.

The second version of MobileNet was published in 2019. The main concept of this second version is to use an inverted residual structure where the shortcut connections are between the thin bottleneck layers.

The current last version of MobileNet was also published 2019. It uses both squeeze and excitation as well as the swish nonlinearity use the sigmoid which can be inefficient to compute as well challenging to maintain accuracy in fixed point arithmetic so we replace this with the hard sigmoid.

![](https://i.imgur.com/awe8Fv0.png)

### ResNet152V2
Theoratically training a single layer is sufficient to represent any function. However it tends to overfit quickly and therefor the way to go in computer vision is going deeper instead. A lot of succesful architectures were built on this idea but there was one problem with it. When creating deeper and deeper CNN networks the gradient can become infinitely small. This is called the vanishing gradient and ResNet[5] is a network built to solve this problem. The network introduces skip connections which are connections that can skip several layers. 

### InceptionResNetV2
InceptionResNet build further upon the inception family. The inception architecture was introduced 2014 [4]. Most networks were focussing on getting deeper and larger CNN networks. However very deep networks are prone to overfitting and it is harder to pass the gradient through the whole network. That is why the researchers of Google introduced Inception network. Instead of going deeper, the authors explore their hypothesis of going 'wider'. The inception block consists of three different filter that are all applied to the same level. Before each filter there is also 1x1 convolution which reduced the numer of features (or channels). The inception network works because the network is less computationally intensive without losing much model complexity and is also less likely to overfit. InceptionResNet using exactly this kind of architecture but combining it with the residual connections introduced in ResNet.

### Transfer learning
For this project there was not a lot of data available. TrashNet contains 2527 images of trash which is nothing compared to for example ImageNet which contains 14 million images. This is why we opted to go for a transfer learning approach for this project. In transfer learning a pretrained model is used which is thus already trained on some dataset. In our case the pretrained models were trained on ImageNet. Altough the network is not specifically trained on garbage data, it can still extract information out of the images. Also the ImageNet classes contains for example plastic bag, trashbin and toilet paper. Altough this is not exactly what we are looking for, we can use these kind of outputs as an input to some new layers. These new layers can learn to map the outcome of a pretrained model to the required class label predictions. Note that only the additional layers are able to update its weights during training. This process is called transfer learning.

# Implementation

## Architecture overview

The first block of our model consists of either MobileNetV3Large, ResNet152V2 or InceptionResNetV2 pre-trained on ImageNet. The top layer is excluded, and only the last six layers are made trainable.

The next layers are mostly based upon [this](https://towardsdatascience.com/advanced-waste-classification-with-machine-learning-6445bff1304f) blog post. Dropout, 2D average pooling and batch normalization are applied to the output of the pre-trained block. The next layer consists of a 256 neuron fully-connected layer, with elu activation and L1 regularization (to avoid overfitting). Finally, dropout is applied again and (expectedly) the last layer is fully-connected and consisting of 5 neurons.

## Libraries and execution

We used Google Colab for training the model and the Keras library was used for implementation.  














## Experiments

### Garbage dataset
Fortunately, a [multitude](https://github.com/AgaMiko/waste-datasets-review) of waste datasets can be found online. They vary greatly in size, number of categories, and origin.   

For our project, we decided to use *[trashnet](https://github.com/garythung/trashnet)* and *[trashbox](https://github.com/nikhilvenkatkumsetty/TrashBox)*. Reason being, that both these datasets mostly overlap in categories and they contain the categories that we require. 

|   | # images | # categories | origin | background
|---|---|---|---|---|
| *trashnet* | 2527 | 6   |  produced |uniform,indoor
| *TrashBox*  | 17785 | 7 | scraped|varying

Furthermore, trashnet images are created by the authors indoors against a uniform background, while trashbox images have been scraped online. We hope that combining these two datasets of different origin will result in our model generalizing well. 

Small changes to the categories had to be made in order to be able to combine the two datasets. The `trash` category has been omitted from trashnet, while `e-waste` and `medical waste` have been omitted from TrashBox. This does not pose a problem, as we are not interested in classyfing these categories. Moreover, trashnet's `trash` category might create problems because it contains a large variety of plastic items.

The 5 remaining categories are: `cardboard`, `glass`, `metal`, `paper` and `plastic`.
|   | Size | Parameters (M) | Depth    | Top-1 Accuracy   |
|---|---|---|---|---|
| MobileNetV3 | 15MB | 5.4   |    | 75.6% | 
| ResNet152V2  | 232MB | 60.4 | 307   | 78.0%   |
| InceptionResNetV2  | 215   | 55.9   | 449   | 80.3%   |

### Different models

The first set of experiments that were done is by comparing different models. Since the focus is to find a suitable model for mobile devices we want to see whether models with a smaller number of learnable parameters can still produce good results compared to models that have a lot of learnable parameters. The size and number of parameters of the models can be found in the table above. Furthermore the graph were done with 200 epochs for each model and were trained and evaluated on the TrashNet dataset.

TrashNet Data:
![](https://i.imgur.com/bgWoi8e.png)             |  ![](https://i.imgur.com/R7Uy0QC.png)
:-------------------------:|:-------------------------:
![](https://i.imgur.com/x2g4ATd.png)  |  

TrashBox Data:
![](https://i.imgur.com/5wTFf98.png)            |  ![](https://i.imgur.com/kS2cces.png)
:-------------------------:|:-------------------------:
![](https://i.imgur.com/03y9MZ8.png)  |  


|   | TrashNet Accuracy | TrashBox Accuracy   |
|---|---|---|
| MobileNetV3 | 0.912 | 0.900 |
| ResNet152V2  | 0.841 |  0.8732  |
| InceptionResNetV2  | 0.875 |  0.882   |

### Data augmentation
As mentioned before we use transfer learning to get good results relatively quickly without having a large dataset. Another way to improve the model and to prevent overfitting data augmentation is often a technique that is used. Therefor, we also tried different data augmentation experiments. Also some research was done to see how this data augmentation really works. The data augmentation layers in keras make sure that each epoch new random data augmentation is done on the original images and these layers are turned of during prediction phase. This makes using data augmentation really easy to use as you just need to put some layers in from of the original model.

By looking at the data it can be seen that it consists of pretty similar conditions in terms of lighting and location. All picture are taken from the same table in the same room. However, pictures are taken from different angles so that is why rotation and flipping seemed as a good first experiment. Also we still wanted to see whether contrast and brightness would make a change. The last plot combines all of these data augmentation layers.


![](https://i.imgur.com/bgWoi8e.png)             |  ![](https://i.imgur.com/Mt1qX4M.png)
:-------------------------:|:-------------------------:
![](https://i.imgur.com/tQCeZfl.png)  |  ![](https://i.imgur.com/yUCPlCu.png)

# Mobile application

The reason behind our decision to implement our model in an Android app is two-fold. Firstly, it serves as a first step towards bringing automated waste classification to the end-user. Secondly, it makes it easier to test our model *in the wild*.

During this project, we wanted to keep our focus on the model itself. We predicted that creating an Android app from scratch could become very time-consuming. Additionally, the two reasons stated above do not require a polished app.

Hence, we decided to look for an open-source Android classification app in which the classification model could be swapped for ours.


Thanks to the *[TensorFlow Lite](https://www.tensorflow.org/lite)* library, it is trivial to convert a TensorFlow model such that it can be deployed on mobile devices. Moreover, MobileNetV3 is tuned to mobile devices, which helps with performance.

The model that we decided to use for the app is MobileNetV3, trained on both `trashnet` and `trashbox` datasets, with random flipping and random rotation, with an L2 regularizer, and for 50 epochs. 

Our reason to use this model in the end, is that we think this model maximizes chances of generalizing succesfully. The datasets vary a lot because pictures of trashbox have very large variance, while pictures of trashnet are taken in a controlled setting.

Data augmentation prevents overfitting. The L2 regularizer allows for very small weights to remain, in contrast to the L1 regularizer which forces small weights towards zero, thus L2 can help with generalization. By training for only 50 epochs, we prevent overfitting

[This](https://github.com/android/camera-samples/) suite of demo apps seemed to be a good candidate, in particular `CameraXAdvanced`. It locally classifies the camera input in real-time into ImageNet categories.

We made the following modifications to the `CameraXApp`:

* **Swapped** out the ImageNet model for our waste model
* Incorporated the waste **categories** into the **label** file
* `CameraXApp` originally expects bounding boxes from the model, which it then draws around the object. We omitted these bounding boxes, as our model doesn't support them.
* Some extra changes in order for input to be of correct shape and dtype.

<img src="https://i.imgur.com/9tPrpCx.gif" width=80px, style="margin-left:5px; margin-right:12px;margin-bottom:10px" />

<img src="https://i.imgur.com/AhVtk4l.gif" width=80px, style="margin-right:12px;margin-bottom:10px" />
<img src="https://i.imgur.com/r6vsZNp.gif" width=80px, style="margin-right:12px;margin-bottom:10px" />
<img src="https://i.imgur.com/7OSsz8m.gif" width=80px, style="margin-right:12px;margin-bottom:10px" />
<img src="https://i.imgur.com/cnWPtNz.gif" width=80px, style="margin-right:12px;margin-bottom:10px" />





The `.apk` executable can be found in `/deliverables`.

# Problems and learning process
First of all there were some problems we ran into during the project. The InceptionResNet and ResNet architectures were not performing at all. The performance of both training and test set were below 50%. By doing some research we found out that the pretrained models can differ in terms of how the images are preprocessed. The same exact preprocessing should be applied when using these pretrained models and after this was implemented everything worked as expected. 

Also we learned a lot about the state-of-the-art computer vision models as for each model that is evaluated we wanted to understand what is different compared to other models.

Lastly, we learned how to get these concepts to work on a domain that was chosen by ourself and to be able to turn it into a real working mobile application.

# Discussion and future work
We feel that we have reached the goal that we had set for ourselves. In the wild, the model classifies almost all 'easy' cases perfectly, under a variety of conditions.

What the model is struggling with, are especially objects that consist of multiple materials. This is to be expected, and a solution should be sought for this. Another problem is that the model has some color bias. Against a white background, the model tends to predict `paper`, even for other materials.

For the experiments it can be seen that MobileNet works really well although the network is much smaller. In fact MobileNet outperforms the other more complex architectures. Both InceptionResNet and ResNet seems to overfit on the data. This is probably due to a combination of the models being more complex and the fact that the dataset that was profided is not big and diverse enough. A lot of images from the dataset were taken in the same room with the same background and the same lighting. We expect that the more complex models will outperform MobileNet when there is a huge amount of diverse data. However it is nice to conclude that MobileNet is a good model that can generalize very well on a small dataset. 

One way to reduce the overfitting behavior is by introducing data augmentation. This is the second experiment that was done. Especially random flip and rotation seems to really bring the training and test scores more close to each other. 

For future work it would be interesting to see whether the complexer models can outperform MobileNet when a huge diverse dataset is provided. Also data augmentation was only done on MobileNet as an experiment and it would be interesting to see this experiment on the complex models as well since there models seems to overfit more.
 
<!-- Problem we ran into:
Some models not working first (preprocessing)
Used soft max but with loss that did not need it
Data seem to not have lot of different environments


Discussion:
Data augentation can make performance worse however more robust in real life.
We put the model into the app to see whether it can work in the realworld (to see if it is robust)
If training is done on trashNet and evaluation on trashBox we go from 0.948 to 0.455 accuracy which means that model is not very robust when only trained on trashnet

Future work:
Combine dataset to make network more robust -->

### References
- [1] Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., Andreetto, M. & Adam, H. (2017). Mobilenets: Efficient convolutional neural networks for mobile vision applications. arXiv preprint arXiv:1704.04861.
- [2] Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). Mobilenetv2: Inverted residuals and linear bottlenecks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 4510-4520).
- [3] Howard, A., Sandler, M., Chu, G., Chen, L. C., Chen, B., Tan, M., Wang, W., Zhu, Y., Pang, R., Vasudevan, V., Le, Q. & Adam, H. (2019). Searching for mobilenetv3. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 1314-1324).
- [4] Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., Erhan, D., Vanhoucke, V. & Rabinovich, A. (2015). Going deeper with convolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1-9).
- [5] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).

