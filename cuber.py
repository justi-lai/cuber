import numpy as np
from cubie import Cubie

class Cube:
    """
    Represents the entire Rubik's Cube as a 3x3x3 grid of Cubie objects.
    The coordinate system is (x, y, z) where:
    - x-axis: 0=Left (O), 1=Middle, 2=Right (R)
    - y-axis: 0=Down (Y), 1=Middle, 2=Up (W)
    - z-axis: 0=Back (B), 1=Middle, 2=Front (G)
    """
    def __init__(self):
        """
        Initializes a 3x3x3 grid of Cubie objects. The initial state of the cube will be solved.
        The coordinate system is (x, y, z) where:
        - x-axis: 0=Left (O), 1=Middle, 2=Right (R)
        - y-axis: 0=Down (Y), 1=Middle, 2=Up (W)
        - z-axis: 0=Back (B), 1=Middle, 2=Front (G)
        """
        self.grid = np.empty((3, 3, 3), dtype=object)

        for x, y, z in np.ndindex(3, 3, 3):
            if x == 1 and y == 1 and z == 1:
                self.grid[x, y, z] = None
                continue
            
            orientation = {}
            if x == 2: orientation["R"] = "R"
            if x == 0: orientation["L"] = "O"
            if y == 2: orientation["U"] = "W"
            if y == 0: orientation["D"] = "Y"
            if z == 2: orientation["F"] = "G"
            if z == 0: orientation["B"] = "B"
            
            self.grid[x, y, z] = Cubie(orientation)

    def show(self):
        """
        Prints a consistent and correctly oriented 2D unfolded representation of the cube.
        The representation will be the standard 2D unfolded representation of the cube.
        """
        FACE_VIEWS = {
            'U': (1, 2, False, False),   'D': (1, 0, True, False),
            'L': (0, 0, True, False),   'R': (0, 2, True, True),
            'F': (2, 2, True, False),   'B': (2, 0, True, True)
        }

        def get_face_map(face_char):
            axis, idx, flip_rows, flip_cols = FACE_VIEWS[face_char]
            
            # Select the correct 2D slice from the 3D grid
            if axis == 0: face_slice = self.grid[idx, :, :]   # L/R faces are (y, z) slices
            elif axis == 1: face_slice = self.grid[:, idx, :] # U/D faces are (x, z) slices
            else: face_slice = self.grid[:, :, idx]           # F/B faces are (x, y) slices

            face_map = np.empty((3, 3), dtype=str)
            
            if face_char in 'UD':
                for (x, z), cubie in np.ndenumerate(face_slice):
                    if cubie: face_map[z, x] = cubie.orientation.get(face_char, ' ')
            else:
                if face_char in 'LR':
                    for (y, z), cubie in np.ndenumerate(face_slice):
                        if cubie: face_map[y, z] = cubie.orientation.get(face_char, ' ')
                else:
                    for (x, y), cubie in np.ndenumerate(face_slice):
                        if cubie: face_map[y, x] = cubie.orientation.get(face_char, ' ')
            
            if flip_rows: face_map = np.flip(face_map, axis=0)
            if flip_cols: face_map = np.flip(face_map, axis=1)
            return face_map

        U, L, F, R, B, D = (get_face_map(f) for f in "ULFRBD")

        pad = " " * 8
        print("\n" + pad + f"\n{pad}".join(" ".join(row) for row in U))
        print("-" * 30)
        for rL, rF, rR, rB in zip(L, F, R, B):
            print(" ".join(rL), "|", " ".join(rF), "|", " ".join(rR), "|", " ".join(rB))
        print("-" * 30)
        print(pad + f"\n{pad}".join(" ".join(row) for row in D) + "\n")

    def turn(self, moves: str):
        """
        Performs a turn on the cube based on your definition.
        The syntax is the standard Rubik's Cube notation.
        
        The moves are defined as follows:
        - U: Up
        - D: Down
        - L: Left
        - R: Right
        - F: Front
        - B: Back
        - M: Middle
        - E: Edge
        - S: Corner
        - x: 90 degree clockwise turn around the x-axis (U)
        - y: 90 degree clockwise turn around the y-axis (L)
        - z: 90 degree clockwise turn around the z-axis (F)

        The modifiers are defined as follows:
        - ': 90 degree counter-clockwise turn
        - 2: 180 degree turn
        - : 90 degree clockwise turn

        The moves are performed in the order they are given.

        For example:
        "U L2 D' L2 F R U D' F' L' B2 R D2 B2 R2 U2 F2 R' F2 R2 F2"

        Args:
            moves (str): The moves to perform on the cube.
        """
        for move in moves.split():
            if len(move) == 0:
                continue

            move_char = move[0]
            modifier = move[1:]
            if move_char not in "UDLRFBudlrfbMESxyz":
                raise ValueError(f"Invalid face '{move_char}' in move '{move_char}'.")

            if modifier == "'":
                times = 3
            elif modifier == "2":
                times = 2
            else:
                times = 1
            
            for _ in range(times):
                self._turn_clockwise(move_char)

    def _turn_clockwise(self, move: str):
        """
        Turns a face clockwise.
        The availablemoves are:
        - U: Up
        - D: Down
        - L: Left
        - F: Front
        - B: Back
        - M: Middle
        - E: Edge
        - S: Corner

        Args:
            move (str): The move to perform.
        """
        if move == "U":
            move_slice = self.grid[:, 2, :].copy()
            for x in range(3):
                for z in range(3):
                    self.grid[2-z, 2, x] = move_slice[x, z]
            for cubie in self.grid[:, 2, :].flatten():
                cubie.turn("U")
                
        elif move == "D":
            move_slice = self.grid[:, 0, :].copy()
            for x in range(3):
                for z in range(3):
                    self.grid[z, 0, 2-x] = move_slice[x, z]
            for cubie in self.grid[:, 0, :].flatten():
                cubie.turn("D")
        
        elif move == "R":
            move_slice = self.grid[2, :, :].copy()
            for y in range(3):
                for z in range(3):
                    self.grid[2, z, 2-y] = move_slice[y, z]
            for cubie in self.grid[2, :, :].flatten():
                cubie.turn("R")
        
        elif move == "L":
            move_slice = self.grid[0, :, :].copy()
            for y in range(3):
                for z in range(3):
                    self.grid[0, 2-z, y] = move_slice[y, z]
            for cubie in self.grid[0, :, :].flatten():
                cubie.turn("L")

        elif move == "F":
            move_slice = self.grid[:, :, 2].copy()
            for x in range(3):
                for y in range(3):
                    self.grid[y, 2-x, 2] = move_slice[x, y]
            for cubie in self.grid[:, :, 2].flatten():
                cubie.turn("F")

        elif move == "B":
            move_slice = self.grid[:, :, 0].copy()
            for x in range(3):
                for y in range(3):
                    self.grid[2-y, x, 0] = move_slice[x, y]
            for cubie in self.grid[:, :, 0].flatten():
                cubie.turn("B")
        
        elif move == "M":
            move_slice = self.grid[1, :, :].copy()
            for y in range(3):
                for z in range(3):
                    self.grid[1, 2-z, y] = move_slice[y, z]
            for cubie in self.grid[1, :, :].flatten():
                if cubie:
                    cubie.turn("M")
        
        elif move == "E":
            move_slice = self.grid[:, 1, :].copy()
            for x in range(3):
                for z in range(3):
                    self.grid[z, 1, 2-x] = move_slice[x, z]
            for cubie in self.grid[:, 1, :].flatten():
                if cubie:
                    cubie.turn("E")
        
        elif move == "S":
            move_slice = self.grid[:, :, 1].copy()
            for x in range(3):
                for y in range(3):
                    self.grid[y, 2-x, 1] = move_slice[x, y]
            for cubie in self.grid[:, :, 1].flatten():
                if cubie:
                    cubie.turn("S")
        
        else:
            raise ValueError(f"Invalid face '{move}' in turn.")


if __name__ == "__main__":
    my_cube = Cube()
    
    print("--- Performing \"U L2 D' L2 F R U D' F' L' B2 R D2 B2 R2 U2 F2 R' F2 R2 F2\" ---")
    my_cube.turn("U L2 D' L2 F R U D' F' L' B2 R D2 B2 R2 U2 F2 R' F2 R2 F2")
    my_cube.show()