import random
import mnist_loader

import torch
import torch.nn as nn


def to_tensors(data, vectorized_targets):
    """Convert mnist_loader's (784,1)/(10,1) numpy column vectors into
    the flat (784,)/(10,) tensors used here."""
    return [
        (torch.from_numpy(x).float().view(-1),
         torch.from_numpy(y).float().view(-1) if vectorized_targets else int(y))
        for x, y in data
    ]


def feedforward(weights, biases, a):
    """Return the output of the network if "a" is input."""
    for w, b in zip(weights, biases):
        a = torch.sigmoid(a @ w.T + b)
    return a

def evaluate(weights, biases, test_data):
    """Return the number of test inputs for which the network
    outputs the correct result."""
    with torch.no_grad():
        results = [(torch.argmax(feedforward(weights, biases, x)), y) for x, y in test_data]
    return sum(int(x == y) for x, y in results)


def main():
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    training_data = to_tensors(list(training_data), vectorized_targets=True)
    test_data = to_tensors(list(test_data), vectorized_targets=False)

    sizes = [784, 30, 10]
    weights = []
    biases = []
    for i in range(1, len(sizes)):
        w = torch.randn(sizes[i], sizes[i-1], requires_grad=True)
        weights.append(w)
        b = torch.rand(sizes[i], requires_grad=True)
        biases.append(b)

    # for w in weights:
    #     print(w.shape)
    #     print(w.grad)
    epochs = 30
    mini_batch_size = 20
    eta = 3.0

    if test_data: n_test = len(test_data)
    n = len(training_data)

    for j in range(epochs):
        random.shuffle(training_data)
        mini_batches = [
            training_data[k:k + mini_batch_size]
            for k in range(0, n, mini_batch_size)
        ]
        for mini_batch in mini_batches:
            mini_batch_loss = 0
            for x,y in mini_batch:
                training_activ = feedforward(weights, biases, x)        # FORWARD PASS
                training_loss = ((training_activ - y) ** 2).sum()       # LOSS
                mini_batch_loss += training_loss                        # LOSS AGGREGATION
            mini_batch_loss = mini_batch_loss / len(mini_batch)         # LOSS AVERAGING
            mini_batch_loss.backward()                                  # BACKWARD PASS
            with torch.no_grad():
                for w in weights:
                    w -= eta * w.grad                                   # WEIGHTS UPDATE FOR NEXT PASS
#                    print(w.grad.shape)
                    w.grad = None                                       # ZEROING GRAD FOR NEXT PASS
                for b in biases:
                    b -= eta * b.grad                                   # BIAS UPDATE FOR NEXT PASS
                    b.grad = None                                       # ZEROING GRAD FOR NEXT PASS
        if test_data:
            print(f"Epoch {j}: {evaluate(weights, biases, test_data)} / {n_test}")
        else:
            print(f"Epoch {j} complete")

if __name__ == "__main__":
    main()