# cuber

A simple, robust, and fully virtual Rubik's Cube implementation in Python.

## Why?

I created this project because I found that [pycuber](https://github.com/adrianliaw/PyCuber) was not working well for my needs and was somewhat buggy. I wanted a more reliable, transparent, and hackable Rubik's Cube simulator that I could easily understand, extend, and use for experiments or educational purposes.

## Features

- **3D Virtual Cube:** Models a standard 3x3x3 Rubik's Cube using a grid of "cubie" objects.
- **Customizable Moves:** Supports all standard face turns (`U`, `D`, `L`, `R`, `F`, `B`), slice moves (`M`, `E`, `S`), and their inverses and double turns.
- **Accurate Orientation:** Each cubie tracks its own orientation, ensuring realistic behavior.
- **Unfolded Display:** Prints a 2D unfolded view of the cube for easy visualization.
- **Minimal Dependencies:** Only requires `numpy` (see `requirements.txt`).

## Usage

```python
from cuber import Cube

cube = Cube()
cube.show()  # Display the solved state

cube.turn("U R2 F'")  # Perform moves
cube.show()           # See the result
```

Or run the script directly to see example moves and their effects.

## File Structure

- `cuber.py` — Main cube logic, move handling, and display.
- `cubie.py` — Individual cubie logic and orientation management.
- `requirements.txt` — Lists required Python packages (just `numpy`).

## Requirements

- Python 3.x
- numpy (see `requirements.txt`)

## Installation

Clone this repo and install the requirements:

```bash
pip install -r requirements.txt
```

## Motivation

I wanted a Rubik's Cube simulator that was:

- **Reliable:** No mysterious bugs or unexpected behavior.
- **Understandable:** Easy to read and modify.
- **Educational:** Good for learning about cube mechanics and programming.

## Current Improvements (Planned/Upcoming)

- More efficient handling of prime (') and double (2) moves
- Support for wide moves (e.g., `Rw`, `Uw`)
- Ability to rotate the entire cube (x, y, z rotations)
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
