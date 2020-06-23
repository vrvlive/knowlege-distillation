import os
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms as T


class ImageDataLoader():
    def __init__(self, hparams):
        self._hparams = hparams
        self._val_transform = T.Compose([
            T.Resize((self._hparams.image_size, self._hparams.image_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self._train_transform = T.Compose([
            T.RandomRotation(30),
            T.RandomResizedCrop((self._hparams.image_size, self._hparams.image_size)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def get_train_dataloader(self):
        data_folder = os.path.join(self._hparams.data_dir, self._hparams.train_data)
        image_folder = ImageFolder(data_folder, transform=self._train_transform)
        return DataLoader(image_folder, batch_size=self._hparams.batch_size,
                          shuffle=True, num_workers=self._hparams.num_workers)

    def get_valid_dataloader(self):
        data_folder = os.path.join(self._hparams.data_dir, self._hparams.val_data)
        image_folder = ImageFolder(data_folder, transform=self._val_transform)
        return DataLoader(image_folder, batch_size=self._hparams.batch_size,
                          shuffle=False, drop_last=True, num_workers=self._hparams.num_workers)

    def get_test_dataloader(self):
        pass
