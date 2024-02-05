import torch
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import Subset


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


def load_dataset(config):
    train_data = torchvision.datasets.CIFAR100('./data', train=True, download=True)
    x = np.concatenate([np.asarray(train_data[i][0]) for i in range(len(train_data))])
    mean = np.mean(x, axis=(0, 1))/255
    std = np.std(x, axis=(0, 1))/255

    transform_t = transforms.Compose([
            transforms.RandomChoice([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(180),
                transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                ]),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    transform_v = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    dataset_load_func = {
        'mnist': torchvision.datasets.MNIST,
        'fmnist':torchvision.datasets.FashionMNIST,
        'cifar10': torchvision.datasets.CIFAR10,
        'cifar100': torchvision.datasets.CIFAR100,
        'imagenet' : torchvision.datasets.ImageFolder
    }

    if config.mode == "train":
        dataset = dataset_load_func[config.dataset]("./data", train=True, download=True, transform=transform_v)
        train_dataset, valid_dataset = train_test_split(dataset, test_size=0.2, shuffle=False)
        
        #train_dataset = Subset(dataset, train_dataset)
        #valid_dataset = Subset(dataset, valid_dataset)
        #train_dataset.dataset.transform = transform_t
        #valid_dataset.dataset.transform = transform_v
        
        train_dataloader = DataLoader(tuple(train_dataset), batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=8)
        valid_dataloader = DataLoader(tuple(valid_dataset), batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=8)
        
        return train_dataloader, valid_dataloader
    else:
        test_dataset = torchvision.datasets.CIFAR100("./data", train=False, download=True, transform=transform_v)
        test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=8)
        return test_dataloader
