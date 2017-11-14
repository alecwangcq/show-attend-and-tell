import torch
import torch.utils.data as data
from PIL import Image
from os import listdir
from os.path import join


class ImageExtractDataset(data.Dataset):

    def __init__(self, root, files=None, transform=None):
        self._root = root
        self._transform = transform
        self._files = self._collect_images() if files is None else files

    def _collect_images(self):
        root = self._root
        files = [f for f in listdir(root)]
        return files

    def __getitem__(self, index):
        root = self._root
        files = self._files
        image = join(root, files[index])
        image = Image.open(image).convert('RGB')
        image = image.resize([224, 224], Image.LANCZOS)
        if self._transform is not None:
            image = self._transform(image)

        return image, files[index]

    def __len__(self):
        return len(self._files)

    def collate_fn(self, data):
        images, names = zip(*data)
        images = torch.stack(images, 0)

        return images, names


def get_loader(root, files, transform, batch_size, shuffle, num_workers):

    dataset = ImageExtractDataset(root, files, transform)
    data_loader = torch.utils.data.DataLoader( dataset=dataset, batch_size=batch_size,
                                               shuffle=shuffle, num_workers=num_workers,
                                               collate_fn=dataset.collate_fn)

    return data_loader







