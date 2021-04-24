from typing import Any, Tuple

import torchvision.datasets as datasets
from tqdm import tqdm


class ImageFolderDatasetPreload(datasets.ImageFolder):
    def __init__(self, root: str, *args, **kwargs):
        super().__init__(root, *args, **kwargs)
        self._preload_data()

    def _preload_data(self) -> None:
        self._loaded_samples = [self.loader(path) for path, _ in tqdm(self.samples)]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        _, target = self.samples[index]
        sample = self._loaded_samples[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
