import pytorch_lightning as pl
import torch as th
from torch import nn, optim

from cnnmodels import CnnModelFactory
from image_loader import ImageDataLoader


class TrainingModule(pl.LightningModule):
    def __init__(self, model_name, hparams, pre_trained=False):
        super(TrainingModule, self).__init__()
        self._hparams = hparams
        self._loss = nn.CrossEntropyLoss()
        self._model = self.create_model(model_name=model_name, pre_trained=pre_trained)
        self._dataloader = None

    def create_model(self, model_name, pre_trained):
        model_factory = CnnModelFactory()
        model = model_factory.create_model(model_name=model_name,
                                           image_size=self._hparams.image_size,
                                           num_classes=self._hparams.num_classes,
                                           pre_trained=pre_trained)
        return model

    def prepare_data(self):
        self._dataloader = ImageDataLoader(hparams=self._hparams)

    def forward(self, images):
        return self._model(images)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        y_hat = self.forward(images)
        loss = self._loss(y_hat, labels).mean()
        return {'loss': loss,
                'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        y_hat = self.forward(images)
        loss = self._loss(y_hat, labels)
        acc = (y_hat.argmax(-1) == labels).float()
        return {'loss': loss,
                'acc': acc}

    def validation_epoch_end(self, outputs):
        loss = th.stack([output['loss'] for output in outputs], dim=0).mean()
        acc = th.stack([output['acc'] for output in outputs], dim=0).mean()
        out = {'val_loss': loss, 'val_acc': acc}
        return {**out, 'log': out}

    def train_dataloader(self):
        return self._dataloader.get_train_dataloader()

    def val_dataloader(self):
        return self._dataloader.get_valid_dataloader()

    def configure_optimizers(self):
        return optim.SGD(self._model.parameters(),
                         lr=self._hparams.learning_rate,
                         momentum=self._hparams.momentum)


class DistilledTrainingModule(TrainingModule):
    def __init__(self, student_model_name, teacher_model_name, hparams):
        super(DistilledTrainingModule, self).__init__(model_name=student_model_name, hparams=hparams)
        self._mse_loss = nn.MSELoss()
        self._teacher_model = self.create_model(self._hparams.teacher_model, pre_trained=True)
        self._teacher_model.load()
        for param in self._teacher_model.parameters():
            param.requires_grad = False

    def training_step(self, batch, batch_idx):
        images, labels = batch
        y_hat_student = self.forward(images)
        y_hat_teacher = self._teacher_model.forward(images)
        loss = self._mse_loss(y_hat_student, y_hat_teacher)
        return {'loss': loss,
                'log': {'train_loss': loss}}
