import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

data_transforms = {
            'train': transforms.Compose([
                transforms.Resize([256, 256]),
                transforms.RandomCrop((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize([256, 256]),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

def get_art():
    tr_dataset = ImageFolder('/PATH/TO/OFFICE-HOME/ART', data_transforms['train'])
    te_dataset = ImageFolder('/PATH/TO/OFFICE-HOME/ART', data_transforms['test'])

    
    train_loader =torch.utils.data.DataLoader(tr_dataset, batch_size=1, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(te_dataset, batch_size=32, shuffle=False, num_workers=16, pin_memory=True, drop_last=True)
    
    return train_loader, val_loader

def get_clipart():
    tr_dataset = ImageFolder('/PATH/TO/OFFICE-HOME/CLIPART', data_transforms['train'])
    te_dataset = ImageFolder('/PATH/TO/OFFICE-HOME/CLIPART', data_transforms['test'])

    
    train_loader =torch.utils.data.DataLoader(tr_dataset, batch_size=1, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(te_dataset, batch_size=32, shuffle=False, num_workers=16, pin_memory=True, drop_last=True)
    
    return train_loader, val_loader

def get_product():
    tr_dataset = ImageFolder('/PATH/TO/OFFICE-HOME/PRODUCT', data_transforms['train'])
    te_dataset = ImageFolder('/PATH/TO/OFFICE-HOME/PRODUCT', data_transforms['test'])

    
    train_loader =torch.utils.data.DataLoader(tr_dataset, batch_size=1, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(te_dataset, batch_size=32, shuffle=False, num_workers=16, pin_memory=True, drop_last=True)
    
    return train_loader, val_loader

def get_real_word():
    tr_dataset = ImageFolder('/PATH/TO/OFFICE-HOME/PRODUCT/REALWORD', data_transforms['train'])
    te_dataset = ImageFolder('/PATH/TO/OFFICE-HOME/PRODUCT/REALWORD', data_transforms['test'])

    
    train_loader =torch.utils.data.DataLoader(tr_dataset, batch_size=1, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(te_dataset, batch_size=32, shuffle=False, num_workers=16, pin_memory=True, drop_last=True)
    
    return train_loader, val_loader
