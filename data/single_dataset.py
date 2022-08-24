import numpy as np
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
import torchvision.transforms as transforms
from PIL import Image


class SingleDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        # Yi made change here so that npz file could be loaded
        self.is_npz = opt.is_npz
        extensions = ['.npz'] if self.is_npz else None
        self.A_paths = sorted(make_dataset(opt.dataroot, opt.max_dataset_size, extensions=extensions))

        input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc

        # Yi made change here so that we only convert the input to tensor
        # if the input are npz files
        if not self.is_npz:
            self.transform = get_transform(opt, grayscale=(input_nc == 1))
        else:
            transform_list = [transforms.ToTensor()]
            self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        A_path = self.A_paths[index]
        # Yi made change here so that npz files could be loaded
        if not self.is_npz:
            A_img = Image.open(A_path).convert('RGB')
        else:
            with np.load(A_path) as f:
                A_img = f[f.files[0]].astype(np.float32)
        A = self.transform(A_img)
        return {'A': A, 'A_paths': A_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
