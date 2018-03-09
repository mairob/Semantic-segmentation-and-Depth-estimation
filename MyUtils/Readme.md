## This folder contains files neccesariy for a comfortable training.

### Global_Variables.json
Used for setting up different Loggin and Source-Folders, number of epochs, batch-sizes, logging intervals and so on.

### ImageProcessing.py
Used for transforming the plain numerical representation of images after inference into more pleasing RGB-Images. 
Also it covers wrappers for reading batches from TFRecords and preprocessing of network inputs

### LayerBuildingBlocks.py
Wrappers for differnt network elements used (Different Modules like Atrous-Pyramids, Global Convolution Module, Up-Projection...)

### Network.py
Main file that contains the whole structural network defintion. It further defines losses and loss-functions, optimizer (normal and with aggregated gradients) and queue-pipelines. Please adapt it with the desired topology defined in ./Networkdefinitions/*.py

