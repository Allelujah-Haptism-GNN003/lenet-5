# Lenet-5

## Use with torch.hub

```python
from torch import hub

model = hub.load('Flursky/lenet-5:master', 'lenet5', pretrained=False)

```

## References:
- _original paper:_ [Gradient-Based Learning Applied to Document Recognition](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)
- _blogs_: [Muhammad Rizwan Article](https://engmrk.com/lenet-5-a-classic-cnn-architecture/)