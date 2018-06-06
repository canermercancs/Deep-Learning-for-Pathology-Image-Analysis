# PathoImage
Framework for Pathology Image Operations with Deep Learning.

### digiPath module includes whole slide image processing operations. -completed
Please see <a href="https://github.com/canermercancs/Deep-Learning-for-Pathology-Image-Analysis/blob/master/PLAYGROUND_IMAGES.ipynb"> PLAYGROUND_IMAGES.ipynb</a> and Patch_Sampling.ipynb notebooks to see a selection of operations.

### deepnets module includes the functionality of finetuning previously trained convnets on breast histopathology images. Alexnet and VGG are supported. -ongoing
Supported operations include: <br>
<ul>
  <li>training convnets from scratch on breast histopathology images.</li>
  <li>finetuning on patches of consensus ROIs of expert pathologists in histopathology images with 2, 4 or 14 class diagnostic labels associated by the pathologists.</li>
  <li>extracting convolutional feature representations of small patches of pathology images from a finetuned convnet.</li>
</ul>
