import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.utils.data

from impl import LeNet5

if __name__ == "__main__":
    preprocess = transforms.Compose([
        transforms.Resize(32),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    LABELS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    testdata = torchvision.datasets.MNIST('./data', train=False, download=True, transform=preprocess)

    testloader = torch.utils.data.DataLoader(testdata, batch_size=1, num_workers=1)

    lenet5 = LeNet5()

    lenet5.load_state_dict(torch.load('./model/lenet5.model'))
    lenet5.eval()

    total = len(testdata)
    correct = 0.0

    for (image, label) in iter(testloader):
        output = lenet5(image)
        _, index = torch.max(output, 1)
        prediction = LABELS[index]
        groundtruth = label.item()

        if prediction == groundtruth:
            correct += 1
    
    acc = (correct / total) * 100

    print('Accuracy: {}%'.format(acc))
    