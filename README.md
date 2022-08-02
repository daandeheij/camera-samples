# TrashCam: Classifying waste using computer vision

<img src="https://i.imgur.com/AhVtk4l.gif" width=80px, style="margin-right:12px;margin-bottom:10px" />

**TrashCam** is a proof-of-concept app, which determines the material of objects in real-time using a camera on a mobile device. 

The app consists of a modified version of the [`CameraXAdvanced`](https://github.com/android/camera-samples/) demo app, and utilizes a MobileNetV3 network trained on modified versions of both [TrashBox](https://github.com/nikhilvenkatkumsetty/TrashBox) and [trashnet](https://github.com/garythung/trashnet).  

- A detailed readme, the `.apk` executable, and python code used to train model and obtain results can be found in directory [/deliverables/](https://github.com/daandeheij/camera-samples/tree/deliverables/deliverables)
- App code can be found in [/CameraXAdvanced/tflite/src/main/java/com/example/android/camerax/tflite](https://github.com/daandeheij/camera-samples/tree/deliverables/CameraXAdvanced/tflite/src/main/java/com/example/android/camerax/tflite)

Modified datasets can be found on Kaggle:

* Modified Trashbox: https://www.kaggle.com/datasets/ddeheij/trashbox-limited/
* Modified Trashnet: https://www.kaggle.com/datasets/ddeheij/trashnet-limited/


<img src="https://i.imgur.com/9tPrpCx.gif" width=80px, style="margin-left:5px; margin-right:12px;margin-bottom:10px" />

<img src="https://i.imgur.com/r6vsZNp.gif" width=80px, style="margin-right:12px;margin-bottom:10px" />
<img src="https://i.imgur.com/7OSsz8m.gif" width=80px, style="margin-right:12px;margin-bottom:10px" />
<img src="https://i.imgur.com/cnWPtNz.gif" width=80px, style="margin-right:12px;margin-bottom:10px" />
