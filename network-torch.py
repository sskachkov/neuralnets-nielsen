import random

import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        self.sizes = sizes
        self.layers = nn.ModuleList([
            nn.Linear(x, y) for x, y in zip(sizes[:-1], sizes[1:])
        ])

    def feedforward(self, a):
        """Return the output of the network if "a" is input."""
        for layer in self.layers:
            a = torch.sigmoid(layer(a))
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """Train the network using mini-batch stochastic gradient
        descent, letting PyTorch's autograd handle backprop."""
        optimizer = torch.optim.SGD(self.parameters(), lr=eta)
        loss_fn = nn.MSELoss(reduction='sum')
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, optimizer, loss_fn)
            if test_data:
                print(f"Epoch {j}: {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {j} complete")

    def update_mini_batch(self, mini_batch, optimizer, loss_fn):
        """Update weights and biases for a single mini batch via
        optimizer.step(), using gradients computed by loss.backward()."""
        optimizer.zero_grad()
        loss = sum(loss_fn(self.feedforward(x), y) for x, y in mini_batch)
        loss = loss / len(mini_batch)
        loss.backward()
        optimizer.step()

    def evaluate(self, test_data):
        """Return the number of test inputs for which the network
        outputs the correct result."""
        with torch.no_grad():
            results = [(torch.argmax(self.feedforward(x)), y) for x, y in test_data]
        return sum(int(x == y) for x, y in results)


def to_tensors(data, vectorized_targets):
    """Convert mnist_loader's (784,1)/(10,1) numpy column vectors into
    the flat (784,)/(10,) tensors nn.Linear expects."""
    return [
        (torch.from_numpy(x).float().view(-1),
         torch.from_numpy(y).float().view(-1) if vectorized_targets else int(y))
        for x, y in data
    ]


if __name__ == "__main__":
    import mnist_loader

    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    training_data = to_tensors(list(training_data), vectorized_targets=True)
    test_data = to_tensors(list(test_data), vectorized_targets=False)

    net = Network([784, 30, 10])
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
