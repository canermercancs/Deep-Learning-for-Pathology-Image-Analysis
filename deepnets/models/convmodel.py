import torch.nn as nn

class ConvnetModel(nn.Module):
    def __init__(self, num_classes, model_name, model):
        super().__init__()
        self.model_name = model_name
        if self.model_name.startswith('alex'):
            self.features   = model.features
            self.classifier = nn.Sequential(
                nn.Dropout(p = 0.5),
                nn.Linear(256*6*6, 4096), # 9216 -> 4096
                nn.ReLU(inplace = True),
                nn.Dropout(p = 0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace = True),
                nn.Linear(4096, num_classes))

        elif self.model_name.startswith('vgg'):
            self.features   = model.features
            self.classifier = nn.Sequential(
                nn.Linear(512*7*7, 4096), # 25088 -> 4096
                nn.ReLU(inplace = True),
                nn.Dropout(p = 0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace = True),
                nn.Dropout(p = 0.5),
                nn.Linear(4096, num_classes),
            )

        #elif self.model_name.startswith('resnet'):
        #   kinda more complicated to do than alex and vgg nets.
        #   will do later on.  
        else:
            raise NotImplementedError
        # freeze the feature extraction parameters.
        for params in self.features.parameters():
            params.requires_grad = False

    def forward(self, inp):
        inp = self.features(inp)
        inp = inp.view(inp.size(0), -1) # flatten the convolutional output to (batch_size x 256*6*6)
        inp = self.classifier(inp)
        return inp


