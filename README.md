# cuber

A simple, robust, and fully virtual Rubik's Cube implementation in Python.

## Why?

I created this project because I found that [pycuber](https://github.com/adrianliaw/PyCuber) was not working well for my needs and was somewhat buggy. I wanted a more reliable, transparent, and hackable Rubik's Cube simulator that I could easily understand, extend, and use for experiments or educational purposes.

## Features

- **3D Virtual Cube:** Models a standard 3x3x3 Rubik's Cube using a grid of "cubie" objects.
- **Customizable Moves:** Supports all standard face turns (`U`, `D`, `L`, `R`, `F`, `B`), slice moves (`M`, `E`, `S`), wide moves (`r`, `l`, `u`, `d`, `f`, `b`), and whole-cube rotations (`x`, `y`, `z`), including their inverses and double turns.
- **Accurate Orientation:** Each cubie tracks its own orientation, ensuring realistic behavior.
- **Unfolded Display:** Prints a 2D unfolded view of the cube for easy visualization.
- **Minimal Dependencies:** Only requires `numpy` (see `requirements.txt`).

## Usage

```python
from cuber import Cube

cube = Cube()
cube.show()  # Display the solved state

cube.turn("U L2 D' L2 F R U D' F' L' B2 R D2 B2 R2 U2 F2 R' F2 R2 F2")  # Perform moves
cube.show()           # See the result
```

Example output of `cube.show()`:

```
        Y B O
        B W R
        W Y B
------------------------------
B O G | R G W | R G G | W Y O
O O B | R G W | R R B | W B W
O G R | Y R O | G G R | B O W
------------------------------
        G Y Y
        O Y W
        B Y Y
```

Or run the script directly to see example moves and their effects.

## File Structure

- `cuber.py` — Main cube logic, move handling, and display.
- `cubie.py` — Individual cubie logic and orientation management.
- `requirements.txt` — Lists required Python packages (just `numpy`).
- `tests/` — Test directory containing comprehensive test suite.
  - `test_cuber.py` — Full test coverage for both Cube and Cubie classes.

## Requirements

- Python 3.x
- numpy (see `requirements.txt`)

## Installation

Clone this repo and install the requirements:

```bash
pip install -r requirements.txt
```

## Testing

The project includes a comprehensive test suite that covers all functionality:

```bash
python tests/test_cuber.py
```

The test suite includes:

- **Cubie Tests:** Initialization, orientation updates, move transformations, and validation
- **Cube Tests:** All move types (basic, prime, double, slice, wide, rotations), sequences, and edge cases
- **Integration Tests:** Complex patterns, move equivalences, and real scramble sequences
- **22 total tests** covering every aspect of the cube implementation

All tests should pass, confirming that:

- ✅ All move types work correctly
- ✅ Move cancellation works (R followed by R' returns to original state)
- ✅ Identity operations work (four quarter turns = original state)
- ✅ Complex move sequences execute without errors
- ✅ Error handling works for invalid moves

## Motivation

I wanted a Rubik's Cube simulator that was:

- **Reliable:** No mysterious bugs or unexpected behavior.
- **Understandable:** Easy to read and modify.
- **Educational:** Good for learning about cube mechanics and programming.

## Current Improvements (Planned/Upcoming)

- More efficient handling of prime (') and double (2) moves
- Direct access to each face's stickers (for inspection or export)
- Initialize a cube directly from a sticker format
- Reverse a set of moves (generate the inverse algorithm)

## Future Objectives

After these improvements, the main objective will be to:

- Implement algorithms to find solutions for Cross, F2L, OLL, and PLL steps of the CFOP method
- Use these solvers to generate training data for a neural network
- Pre-train the model on generated data, then fine-tune it using top cuber solves
- Develop a CFOP solver that imitates top solvers, aiming for ergonomic and efficient solutions
- Provide example solves to help cubers train and learn from high-quality, human-like solutions
