import torchvision
import torchvision.transforms as transforms
from template import Template

if __name__ == '__main__':
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        )
    ])
    trainset = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform = transform)
    validset = torchvision.datasets.CIFAR10(root = './data', train = False, download = True, transform = transform)
    model = Template(class_num = 10, train_dataset = trainset, valid_dataset = validset)
    model.train_and_valid()
    #predict = model.test(testset)