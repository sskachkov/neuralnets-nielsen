import random

import torch


def init_params(sizes):
    weights = [torch.randn(y, x, requires_grad=True) for x, y in zip(sizes[:-1], sizes[1:])]
    biases = [torch.randn(y, requires_grad=True) for y in sizes[1:]]
    return weights, biases


def feedforward(weights, biases, a):
    """Return the output of the network if "a" is input."""
    for w, b in zip(weights, biases):
        a = torch.sigmoid(a @ w.T + b)
    return a


def SGD(weights, biases, training_data, epochs, mini_batch_size, eta, test_data=None):
    """Train the network using mini-batch stochastic gradient
    descent, letting PyTorch's autograd handle backprop."""
    if test_data: n_test = len(test_data)
    n = len(training_data)
    for j in range(epochs):
        random.shuffle(training_data)
        mini_batches = [
            training_data[k:k + mini_batch_size]
            for k in range(0, n, mini_batch_size)
        ]
        for mini_batch in mini_batches:
            update_mini_batch(weights, biases, mini_batch, eta)
        if test_data:
            print(f"Epoch {j}: {evaluate(weights, biases, test_data)} / {n_test}")
        else:
            print(f"Epoch {j} complete")


def update_mini_batch(weights, biases, mini_batch, eta):
    """Update weights and biases for a single mini batch using
    gradients computed by loss.backward()."""
    loss = sum(((feedforward(weights, biases, x) - y) ** 2).sum() for x, y in mini_batch)
    loss = loss / len(mini_batch)
    loss.backward()
    with torch.no_grad():
        for w in weights:
            w -= eta * w.grad
            w.grad = None
        for b in biases:
            b -= eta * b.grad
            b.grad = None


def evaluate(weights, biases, test_data):
    """Return the number of test inputs for which the network
    outputs the correct result."""
    with torch.no_grad():
        results = [(torch.argmax(feedforward(weights, biases, x)), y) for x, y in test_data]
    return sum(int(x == y) for x, y in results)


def to_tensors(data, vectorized_targets):
    """Convert mnist_loader's (784,1)/(10,1) numpy column vectors into
    the flat (784,)/(10,) tensors used here."""
    return [
        (torch.from_numpy(x).float().view(-1),
         torch.from_numpy(y).float().view(-1) if vectorized_targets else int(y))
        for x, y in data
    ]


def main():
    import mnist_loader

    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    training_data = to_tensors(list(training_data), vectorized_targets=True)
    test_data = to_tensors(list(test_data), vectorized_targets=False)

    weights, biases = init_params([784, 30, 10])
    SGD(weights, biases, training_data, 30, 10, 3.0, test_data=test_data)


if __name__ == "__main__":
    main()
