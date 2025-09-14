# Tiny C++ Micrograd MLP

A minimal C++20 implementation of a scalar-based automatic differentiation engine (micrograd style) plus a small multi-layer perceptron (MLP) for toy experiments. Inspired directly by Andrej Karpathy's original Python [micrograd](https://github.com/karpathy/micrograd) and informed by existing C++ ports:

- https://github.com/iyasinn/micrograd-plusplus
- https://github.com/10-zin/cpp-micrograd

This project recreates the core concepts: a `Value` node representing a scalar with a data field, gradient, list of previous nodes, an operation tag, and a lazily defined backward closure. On top of that it builds `Neuron`, `Layer`, and `MLP` abstractions similar in flavor to a micro PyTorch-like API (very small subset).

## Features

- Scalar reverse-mode autograd (dynamic computation graph)
- Basic ops: +, -, unary -, \*, / (via power), pow, tanh, exp
- Gradient accumulation with topological backward pass
- Simple graph visualization to DOT / PNG (requires Graphviz `dot` in PATH)
- Tiny MLP composed of `Neuron` -> `Layer` -> `MLP`
- Training loop example with squared error loss
- Gradient statistics printing to help diagnose saturation / learning rate issues

## Non-Goals / Limitations

- No broadcasting tensors (pure scalars)
- No GPU / parallelization
- No optimizers beyond manual SGD update (you can easily add momentum or Adam)
- No serialization / checkpointing yet
- Not thread-safe (shared mutable graph structures)

## Directory Layout

```
CMakeLists.txt        # Build configuration
Value.h / value.cpp   # Autograd Value implementation
nn.h                  # Module / Neuron / Layer / MLP definitions (header-only style)
main.cpp              # Example training loop
README.md             # This file
```

`nn.cpp` is currently empty; logic lives inline in `nn.h` for simplicity.

## Running & Sample Output

Example (truncated):

```
Number of parameters: 41
Epoch 0 loss: 3.37133
Grad stats: max=7.61118, mean=0.921667, tiny_fraction=0
...
Epoch 19 loss: 0.029236
Grad stats: max=0.0893461, mean=0.0224708, tiny_fraction=0
```

The gradients & loss progression help diagnose if activations saturate (e.g., tanh outputs near ±1 slowing convergence).

## Visualization

After the last epoch the code can emit a DOT + PNG graph of the final loss expression:

```
loss->visualize("mlp_graph.dot");
```

Requires Graphviz:

```
brew install graphviz      # macOS (Homebrew)
# or
sudo apt-get install graphviz
```

Generates `mlp_graph.dot.dot` (DOT) and `mlp_graph.dot.png` (PNG) depending on naming.

### Sample Graph

Below is an example of the rendered computation graph (values and gradients at the time of capture):

![Computation Graph](mlp_graph.dot.png)

## Design Notes

- Each operation allocates a new `Value` with references (shared_ptr) to its predecessors; backward closures capture weak_ptrs to avoid reference cycles.
- Backward constructs a topological ordering once per call.
- After backward you can optionally free graph structure (controlled by flags) for memory reuse.
- `Neuron` initializes weights with `uniform(-1,1)`; you may adapt to Xavier/Glorot to reduce tanh saturation.

## Credits / Inspiration

- Andrej Karpathy’s original micrograd (Python): https://github.com/karpathy/micrograd
- C++ translation references consulted:
  - micrograd-plusplus: https://github.com/iyasinn/micrograd-plusplus
  - cpp-micrograd: https://github.com/10-zin/cpp-micrograd

These repositories shaped the API style and some structural decisions (reverse-mode pattern, operator overloading approach, parameter organization).
