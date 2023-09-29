# DL Template for Image Classification

* 1. [Clone Repo](#CloneRepo)
* 2. [Example Code](#ExampleCode)
* 3. [Dataset Requierment](#DatasetRequierment)
	* a. [Training](#Training)
	* b. [Testing](#Testing)
	* c. [A Sample Dataset Code](#ASampleDatasetCode)
* 4. [Tutorial](#Tutorial)
	* a. [Overview](#Overview)
	* b. [ParamaterS](#ParamaterS)

##  1. <a name='CloneRepo'></a>Clone Repo
```
git clone https://github.com/Lalalalex/DL-Template.git
```

##  2. <a name='ExampleCode'></a>Example Code
```python
import torchvision
import torchvision.transforms as transforms
from template import template

transform = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )
])
# Here use CIFAR10 for example, you can change your own dataset.
trainset = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform = transform)
validset = torchvision.datasets.CIFAR10(root = './data', train = False, download = True, transform = transform)

model = Template(class_num = 10, train_dataset = trainset, valid_dataset = validset)
model.train_and_valid()
#predict = model.test(testset)
```

##  3. <a name='DatasetRequierment'></a>Dataset Requierment
###  3.1. <a name='Training'></a>Training
Dataset should output [image, label].
```python
class Dataset(Dataset):
    def __getitem__(self, index):
        return images[index], labels[index]
```
###  3.2. <a name='Testing'></a>Testing
Dataset should output image.
```python
class Dataset(Dataset):
    def __getitem__(self, index):
        return images[index]
```

###  3.3. <a name='ASampleDatasetCode'></a>A Sample Dataset Code
```python
class Dataset(Dataset):
    def __init__(self, df, is_test_model = False, transforms = None):
        self.df = df
        self.is_test_model = is_test_model
        self.transforms = transforms
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        image = self.get_image(df['path'][index])
        if self.transforms:
            image = self.transforms(image = image)['image']
        if self.is_test_model:
            return image
        label = self.df.iloc[index]['label']
        return image, label
    
    def get_image(image_path):
        image = cv2.imread(image_path)
        image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image_RGB
```

##  4. <a name='Tutorial'></a>Tutorial
###  4.1. <a name='Overview'></a>Overview
```python
class Template:
    def __init__(self, class_num: int, train_dataset: Dataset, valid_dataset: Dataset, 
                 lr = 5e-4, batch_size = 16, num_workers = 4, drop_last = True, seed = 'seed', set_seed = True,
                 model = 'HarDNet68', loss_function = 'Cross Entropy', optimizer = 'AdamW',
                 pretrained = True, save_best = True, use_ttach = False,
                 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

    def set_seed(self, seed):
    
    def set_model(self, model_name, pretrained):
        return model.to(self.device)
        # Intialize model and return, for setting self.model
    
    def get_model(self):
        return self.model
        # Return self.model
        
    def get_model_list(self):
        print('Acceptable models: ', model_list)
        # Print avilable models

    def set_optimizer(self, optimizer_name, lr):
        return optimizer_list[optimizer_name]
        
    def get_optimizer(self):
        return self.optimizer
    
    def get_optimizer_list(self):
        print('Acceptable optimizers: ', optimizer_list)
    
    def set_loss_function(self, loss_function_name):
        return loss_function_list[loss_function_name]
        
    def get_loss_function(self):
        return self.loss_function
    
    def get_loss_function_list(self):
        print('Acceptable loss functions: ', loss_function_list)

    def train_epoch(self):
        # Train self.model one epoch
    def valid_epoch(self, best_accuracy, best_loss):
        return best_accuracy, best_loss
        # Valid self.model one epoch
    
    def train_and_valid(self, epoch = 40):
        if self.save_best:
            return self.best_model
        else:
            return self.model
        # Train and valid self.model all epoch.
        
    def test(self, dataloader):
        return predict_list
        # Return predict results

```

###  4.2. <a name='ParamaterS'></a>Paramaters
- **class_num: int**
    - The number of class to classifier
- **train_dataset: Dataset**
    - Your training dataset
- **valid_dataset: Dataset**
    - Your valid dataset
- lr: float
    - Learning rate, default = 5e-4
- batch_size: int
    - Batch size, default = 16
- num_workers: int
    - The number of workers using in dataloader, default = 4
- drop_last: Boolean
    - If drop the last data could not be a batch in dataloader, default = False
- seed: string
    - Random seed, could be a string, default = 'seed'
- set_seed: Boolean
    - If set the fixed seed or not, default = True
- model: string
    - The name of the model to use, default = 'HarDNet68'
    - Available model: 'ResNet18', 'ResNet50', 'ResNet152', 'HarDNet68', 'HarDNet85'
- loss_function: string
    - The name of the loss function to use, default = 'Cross Entropy'
    - Available loss function: 'Cross Entropy', 'MSE'
- optimizer: string
    - The name of the optimizer to use, default = 'AdamW'
    - Available optimizer: 'SGD', 'Adam', 'AdamW'
- pretrained: Boolean
    - If use pretrained or not, default = True
    - Warining: If using HarDNet with cpu, it could not be pretrained
- save_best: Boolean
    - If save the best model or not, default = True
    - If this is True, than it will save the best model name 'model.pkl'. And the return of train_and_valid() will also be the best model.
    - You can also get the best model by template.best_model
- use_ttach: Boolean
    - If use test time augmentation or not, default = False
    - The ttach library is from https://github.com/qubvel/ttach
    - Using d4_transform() to do ttach
- device: torch.device
    - The device to use, default = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
