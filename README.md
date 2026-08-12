# neuralnet-nielsen

Michael Nielsen's [*Neural Networks and Deep
Learning*](http://neuralnetworksanddeeplearning.com/)

## Setup

Dependencies (Python >= 3.13, numpy, torch) are managed with
[uv](https://docs.astral.sh/uv/):

```sh
uv sync
```


Training data can be fetched from book's own repo:

```sh
mkdir -p data
wget -O data/mnist.pkl.gz https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/mnist.pkl.gz
```

## Implementations

All four train the same `[784, 30, 10]` sigmoid network on MNIST for 30
epochs, printing test-set accuracy after each epoch (`Epoch N: x / 10000`).
A full run takes a few minutes per implementation.

- **`network.py`** — the book's original: sigmoid neurons, quadratic cost,
  hand-written backprop, plain NumPy.
  ```sh
  uv run python network.py
  ```

- **`network-torch.py`** — PyTorch port using `nn.Module`, `nn.Linear`
  layers, `torch.optim.SGD`, and `nn.MSELoss`. Most idiomatic PyTorch of
  the three — no manual backprop or manual weight updates.
  ```sh
  uv run python network-torch.py
  ```

- **`network-torch-functional.py`** — PyTorch port written as plain
  functions (`init_params`, `feedforward`, `SGD`, `update_mini_batch`)
  instead of a class. Autograd computes the gradients, but the weight
  update is written out by hand (`w -= eta * w.grad`), matching the
  book's own update rule.
  ```sh
  uv run python network-torch-functional.py
  ```

- **`network-torch2.py`** — an earlier, rougher draft of the functional
  approach above: everything is inlined in `main()` instead of factored
  into functions, and it still has leftover debug `print()` calls, so
  expect noisy output.
  ```sh
  uv run python network-torch2.py
  ```

