# Training with Complementary Labels

Okay, let's start with loading some necessary libraries

```python
# Import required modules
import torchvision.transforms as transforms
import torch
```
 Now What we need for complementary label learning.

**Note:** The ComplementaryLoss is a wrapper around different methods. Currently it offers
PC, Forward, Gradient, Non-negative Loss and Assumption-Free Methods. Both these losses
can be instianted independently. If you're interested take a look at cll/algo.py.


```python
from cllcontra.models import MLP, LinearModel
from cllcontra.losses import complementary_forward_fn, ComplementaryLoss
```
 And now our Trainer: We only need CLLTrainer. Which is just our general Trainer but adjusted
 to complementary label learning => Train with a complementary dataset and evaluate our model on ordinary train and test datasets.


```python
from cllcontra.trainer import Trainer, CLLTrainer
```

Let's define some hyperparameters.

```python
from types import SimpleNamespace
args = SimpleNamespace(
    weight_decay=1e-4, 
    epochs = 1000, 
    method = 'pc',# 'nn', 'free', 'pc', 'forward'
    batch_size = 256, 
    learning_rate = 5e-5
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```
Go ahead and get your dataset and feed it into our handler built for complementary labels. The Handler generates class priors,
complementary labels, and the necessary loaders.

```python
import torchvision.datasets as dsets
from cllcontra.data import LoaderWithComplementaryLabels
# Define train and test datasets
train_dataset = dsets.MNIST(root='./data/mnist', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = dsets.MNIST(root='./data/mnist', train=False, transform=transforms.ToTensor())

# Initialize DatasetWithComplementaryLabels
data = LoaderWithComplementaryLabels(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    batch_size=args.batch_size  # Specify batch size
)

# Get the loaders and class prior probabilities
ordinary_train_loader, complementary_train_loader, test_loader, ccp = data.get_loaders()
```

Check the number of classes

```python
K = len(train_dataset.classes)
```

Initialize your model, feed everything into the Trainer and fit your model. All the best!!!

```python
model = MLP(input_dim=28*28, hidden_dim=500, output_dim=K)
model = model.to(device)
optimizer = torch.optim.Adam(
    model.parameters(), weight_decay=args.weight_decay, 
    lr=args.learning_rate
)

trainer = CLLTrainer(
    model = model, 
    train_loader= complementary_train_loader, 
    val_loader= None, 
    optimizer=optimizer,
    loss_fn= ComplementaryLoss(K, ccp, args.method), 
    classification=True, 
    device= device, 
    forward_fn= lambda model, inputs, labels, loss_fn, device: complementary_forward_fn(
        model, inputs, labels, loss_fn, ccp=ccp, method=args.method, device=device
    ),
    save_freq = 30
    
)


trainer.fit(args.epochs, train_loader = ordinary_train_loader, test_loader = test_loader)
```

There you go!

