# IkomiaPluginsPython

![](https://ikomia.com/static/showcase/img/home/plugin.png)

Python plugins for Ikomia platform:

- [Colorful Image Colorization](https://github.com/Ikomia-dev/IkomiaPluginsPython/tree/master/ColorfulImageColorization): automatic colorization
- [Covid-Net](https://github.com/Ikomia-dev/IkomiaPluginsPython/tree/master/CovidNet): COVID-19 diagnosis predictions from chest radiography images. **For research purpose only**.
- [Emotion FER Plus](https://github.com/Ikomia-dev/IkomiaPluginsPython/tree/master/EmotionFERPlus): facial emotion recognition
- [Neural Style Transfer](https://github.com/Ikomia-dev/IkomiaPluginsPython/tree/master/NeuralStyleTransfer): render images with painted style of reference pictures (from famous artists)
- [3D ResNet Action Recognition](https://github.com/Ikomia-dev/IkomiaPluginsPython/tree/master/ResNetActionRecognition): human action recognition on videos or camera streams
- [YOLACT](https://github.com/Ikomia-dev/IkomiaPluginsPython/tree/master/Yolact): instance segmentation method with high quality mask

All these plugins are available in the Ikomia Store and can be installed in one-click with the Ikomia software. Users can then reproduce and evaluate each algorithm easily and quickly.

### How to use
As is, Ikomia plugins code can't be executed in a simple Python environment. You will need to add some extra code to convert it into Python script and manage input/output data.

Ikomia plugins are created for the Ikomia platform, they are fully and directly functional with no code in the Ikomia software. Here are the steps to use Ikomia:

1. Create Ikomia account for free [here](https://ikomia.com/accounts/signup/) (if you don't have one)
2. Download [Ikomia software](https://ikomia.com/en/download) and install it (simple wizard installer)
3. Launch the software and log in with your credentials
4. Open the Ikomia Store window, search for your algorithm and install it with the corresponding button
5. Open your input data (images, videos or cameras)
6. In the process pane, search for your new installed plugin and apply it.
7. Enjoy the results!

### External resources
Some of Ikomia plugins need external resources like pre-trained models. In this case, plugin directory contains the model directly or a text file (download_model.txt) including the URL to download it.
