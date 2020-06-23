from collections import OrderedDict

from torch import nn
import torch.nn.functional as F
from torchvision import models
import torch as th


class CnnClassifier(nn.Module):
    def __init__(self, image_size, num_classes, pre_trained=False, name='student'):
        super(CnnClassifier, self).__init__()
        self._state_dict_file = f'{name}_{num_classes}.pth'
        self._image_size = image_size
        self._num_classes = num_classes
        self._pre_trained = pre_trained

    def get_fclayer_list(self, hidden_layers):
        input_layers, output_layers = hidden_layers[:-1], hidden_layers[1:]
        layers = []
        for i, (l1, l2) in enumerate(zip(input_layers, output_layers)):
            layers.append((f'fc{i}', nn.Linear(l1, l2)))
            layers.append((f'relu{i}', nn.ReLU()))

        layers.append(('fc_out', nn.Linear(output_layers[-1], self._num_classes)))
        return layers

    def forward(self, x):
        return self.get_model().forward(x)

    def get_model(self):
        pass

    def save(self):
        th.save(self.get_model().state_dict(), self._state_dict_file)

    def load(self):
        state_dict = th.load(self._state_dict_file)
        self.get_model().load_state_dict(state_dict)


class SimpleClassifier(CnnClassifier):
    def __init__(self, image_size, num_classes, pre_trained):
        super(SimpleClassifier, self).__init__(image_size=image_size, num_classes=num_classes,
                                               pre_trained=pre_trained, name='simple')
    
        self._conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self._conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self._conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self._pool = nn.MaxPool2d(2, 2)
        self._fc1 = nn.Linear(64 * 28 * 28, 512)
        self._fc2 = nn.Linear(512, self._num_classes)
        self._dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self._pool(F.relu(self._conv1(x)))
        x = self._pool(F.relu(self._conv2(x)))
        x = self._pool(F.relu(self._conv3(x)))

        # flatten image input
        x = x.view(-1, 64 * 28 * 28)
        x = self._dropout(x)
        x = F.relu(self._fc1(x))

        # add dropout layer
        x = self._dropout(x)
        x = self._fc2(x)
        return x

    def get_model(self):
        return self


class Resnet18Classifier(CnnClassifier):
    def __init__(self, image_size, num_classes, pre_trained):
        super(Resnet18Classifier, self).__init__(image_size=image_size, num_classes=num_classes,
                                                 pre_trained=pre_trained, name='resnet18')
        self._model = models.resnet18(pretrained=self._pre_trained)
        self._model.fc = nn.Sequential(OrderedDict(self.get_fclayer_list([512, 100])))

    def get_model(self):
        return self._model


class VGG16Classifier(CnnClassifier):
    def __init__(self, image_size, num_classes, pre_trained):
        super(VGG16Classifier, self).__init__(image_size=image_size, num_classes=num_classes, pre_trained=pre_trained,
                                              name='vgg16')
        self._model = models.vgg16(pretrained=self._pre_trained)
        self._model.classifier = nn.Sequential(OrderedDict(self.get_fclayer_list([512 * 7 * 7, 4096, 4096, 2048, 512])))
        # import IPython;IPython.embed();exit(1)

    def get_model(self):
        return self._model


class Densenet121Classifier(CnnClassifier):
    def __init__(self, image_size, num_classes, pre_trained):
        super(Densenet121Classifier, self).__init__(image_size=image_size, num_classes=num_classes, pre_trained=pre_trained,
                                                    name='densenet121')
        self._model = models.densenet121(pretrained=self._pre_trained)
        self._model.classifier = nn.Sequential(OrderedDict(self.get_fclayer_list([1024, 500])))

    def get_model(self):
        return self._model


class CnnModelFactory:
    def __init__(self):
        self._model_builders = {
            'resnet18': Resnet18Classifier,
            'vgg16': VGG16Classifier,
            'densenet121': Densenet121Classifier,
            'simple': SimpleClassifier
        }

    def create_model(self, model_name, **kwargs):
        model_builder = self._model_builders[model_name]
        if not model_builder:
            raise ValueError(f'Model not found - {model_name}')
        return model_builder(**kwargs)


