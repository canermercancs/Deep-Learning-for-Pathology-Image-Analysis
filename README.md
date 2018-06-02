# PathoImage
Framework for Pathology Image Operations with Deep Learning.

## digiPath module includes whole slide image processing operations. (DONE)
-- please see PLAYGROUND_IMAGES.ipynb and Patch_Sampling.ipynb notebooks to see a selection of operations.

## deepnets module includes the functionality of finetuning previously trained convnets on breast histopathology images. Alexnet and VGG are supported. (ONGOING)
Supported operations include:
-- training convnets from scratch on breast histopathology images.
-- finetuning on patches of consensus ROIs of expert pathologists in histopathology images with 2, 4 or 14 class diagnostic labels associated by the pathologists. 
-- extracting convolutional feature representations of small patches of pathology images from a finetuned convnet.

Work in progress.
