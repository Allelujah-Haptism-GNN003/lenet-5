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

    traindata = torchvision.datasets.MNIST('./data', train=True, download=True, transform=preprocess)

    trainloader = torch.utils.data.DataLoader(traindata, batch_size=5, num_workers=2)

    lenet5 = LeNet5()
    criterion = nn.CrossEntropyLoss()
    sgd = optim.SGD(lenet5.parameters(), lr=0.02, momentum=0.9)

    for epoch in range(5):

        for index, (images, labels) in enumerate(trainloader):
            sgd.zero_grad()

            outputs = lenet5(images)
        
            loss = criterion(outputs, labels)
            
            loss.backward()
            sgd.step()

            if index % 5000 == 4999:
                print('epoch: {}, iteration: {}, loss: {}'.format((epoch+1), index, loss.item()))
    torch.save(lenet5.state_dict(), './model/lenet5.model')