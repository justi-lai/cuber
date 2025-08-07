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
        Performs a sequence of turns on the cube using standard Rubik's Cube notation.
        
        The moves are performed in the order they are given, separated by spaces.
        
        Basic Face Moves:
        - U: Up face clockwise
        - D: Down face clockwise
        - L: Left face clockwise
        - R: Right face clockwise
        - F: Front face clockwise
        - B: Back face clockwise
        
        Slice Moves:
        - M: Middle slice (between L and R, follows L direction)
        - E: Equatorial slice (between U and D, follows D direction)  
        - S: Standing slice (between F and B, follows F direction)
        
        Wide Moves:
        - r: Wide right (R + middle slice)
        - l: Wide left (L + middle slice)
        - u: Wide up (U + middle slice)
        - d: Wide down (D + middle slice)
        - f: Wide front (F + middle slice)
        - b: Wide back (B + middle slice)
        
        Cube Rotations:
        - x: Rotate entire cube around x-axis (like R)
        - y: Rotate entire cube around y-axis (like U)
        - z: Rotate entire cube around z-axis (like F)

        Modifiers:
        - ' (prime): Counter-clockwise turn (e.g., U', R', x')
        - 2: 180-degree turn (e.g., U2, R2, x2)
        - (no modifier): Clockwise turn (e.g., U, R, x)

        Examples:
        - "R U R' U'": Basic sequence (sexy move)
        - "U L2 D' L2 F R U D' F' L' B2 R D2 B2 R2 U2 F2 R' F2 R2 F2": Complex scramble
        - "r U r' F R F'": Wide move sequence
        - "x y z": Cube rotation sequence

        Args:
            moves (str): Space-separated string of moves to perform on the cube.
        """
        for move in moves.split():
            if len(move) == 0:
                continue

            move_char = move[0]
            modifier = move[1:]
            if move_char not in "UDLRFBudlrfbMESxyz":
                raise ValueError(f"Invalid face '{move_char}' in move '{move_char}'.")

            if modifier == "'":
                self._turn_counter_clockwise(move_char)
            elif modifier == "2":
                self._turn_double(move_char)
            else:
                self._turn_clockwise(move_char)
            
    def _turn_clockwise(self, move: str):
        """
        Turns a face or performs a cube rotation clockwise (90 degrees).
        
        The available moves are:
        
        Basic Face Moves:
        - U: Up face
        - D: Down face
        - L: Left face
        - R: Right face
        - F: Front face
        - B: Back face
        
        Slice Moves:
        - M: Middle slice (between L and R, follows L direction)
        - E: Equatorial slice (between U and D, follows D direction)
        - S: Standing slice (between F and B, follows F direction)
        
        Wide Moves:
        - r: Wide right (R + M)
        - l: Wide left (L + middle slice)
        - u: Wide up (U + middle slice)
        - d: Wide down (D + middle slice)
        - f: Wide front (F + middle slice)
        - b: Wide back (B + middle slice)
        
        Cube Rotations:
        - x: Rotate entire cube around x-axis (like R)
        - y: Rotate entire cube around y-axis (like U)
        - z: Rotate entire cube around z-axis (like F)

        Args:
            move (str): The move to perform (single character).
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
        
        elif move == "r":
            move_slices = self.grid[1:3, :, :].copy()
            for x_offset in range(2):
                x_coord = 1 + x_offset
                for y in range(3):
                    for z in range(3):
                        self.grid[x_coord, z, 2-y] = move_slices[x_offset, y, z]
            for cubie in self.grid[1, :, :].flatten():
                if cubie:
                    cubie.turn("R")
            for cubie in self.grid[2, :, :].flatten():
                cubie.turn("R")

                
        elif move == "l":
            move_slices = self.grid[0:2, :, :].copy()
            for x_offset in range(2):
                x_coord = x_offset
                for y in range(3):
                    for z in range(3):
                        self.grid[x_coord, 2-z, y] = move_slices[x_offset, y, z]
            for cubie in self.grid[0, :, :].flatten():
                cubie.turn("L")
            for cubie in self.grid[1, :, :].flatten():
                if cubie:
                    cubie.turn("L")
        
        elif move == "u":
            move_slices = self.grid[:, 1:3, :].copy()
            for y_offset in range(2):
                y_coord = 1 + y_offset
                for x in range(3):
                    for z in range(3):
                        self.grid[2-z, y_coord, x] = move_slices[x, y_offset, z]
            for cubie in self.grid[:, 1, :].flatten():
                if cubie:
                    cubie.turn("U")
            for cubie in self.grid[:, 2, :].flatten():
                cubie.turn("U")
        
        elif move == "d":
            move_slices = self.grid[:, 0:2, :].copy()
            for y_offset in range(2):
                y_coord = y_offset
                for x in range(3):
                    for z in range(3):
                        self.grid[z, y_coord, 2-x] = move_slices[x, y_offset, z]
            for cubie in self.grid[:, 0, :].flatten():
                cubie.turn("D")
            for cubie in self.grid[:, 1, :].flatten():
                if cubie:
                    cubie.turn("D")
        
        elif move == "f":
            move_slices = self.grid[:, :, 1:3].copy()
            for z_offset in range(2):
                z_coord = 1 + z_offset
                for x in range(3):
                    for y in range(3):
                        self.grid[y, 2-x, z_coord] = move_slices[x, y, z_offset]
            for cubie in self.grid[:, :, 1].flatten():
                if cubie:
                    cubie.turn("F")
            for cubie in self.grid[:, :, 2].flatten():
                cubie.turn("F")
        
        elif move == "b":
            move_slices = self.grid[:, :, 0:2].copy()
            for z_offset in range(2):
                z_coord = z_offset
                for x in range(3):
                    for y in range(3):
                        self.grid[2-y, x, z_coord] = move_slices[x, y, z_offset]
            for cubie in self.grid[:, :, 0].flatten():
                cubie.turn("B")
            for cubie in self.grid[:, :, 1].flatten():
                if cubie:
                    cubie.turn("B")

        elif move == "x":
            move_slices = self.grid[:, :, :].copy()
            for x in range(3):
                for y in range(3):
                    for z in range(3):
                        self.grid[x, z, 2-y] = move_slices[x, y, z]
            for cubie in self.grid.flatten():
                if cubie:
                    cubie.turn("X")
        
        elif move == "y":
            move_slices = self.grid[:, :, :].copy()
            for x in range(3):
                for y in range(3):
                    for z in range(3):
                        self.grid[2-z, y, x] = move_slices[x, y, z]
            for cubie in self.grid.flatten():
                if cubie:
                    cubie.turn("Y")
        
        elif move == "z":
            move_slices = self.grid[:, :, :].copy()
            for x in range(3):
                for y in range(3):
                    for z in range(3):
                        self.grid[y, 2-x, z] = move_slices[x, y, z]
            for cubie in self.grid.flatten():
                if cubie:
                    cubie.turn("Z")
        
        else:
            raise ValueError(f"Invalid face '{move}' in turn.")

    def _turn_counter_clockwise(self, move: str):
        """
        Turns a face or performs a cube rotation counter-clockwise (90 degrees).
        
        This method performs the inverse of the corresponding clockwise move.
        All moves supported by _turn_clockwise are also supported here in their
        counter-clockwise (prime) form.
        
        The available moves are:
        
        Basic Face Moves:
        - U: Up face counter-clockwise
        - D: Down face counter-clockwise
        - L: Left face counter-clockwise
        - R: Right face counter-clockwise
        - F: Front face counter-clockwise
        - B: Back face counter-clockwise
        
        Slice Moves:
        - M: Middle slice counter-clockwise
        - E: Equatorial slice counter-clockwise
        - S: Standing slice counter-clockwise
        
        Wide Moves:
        - r: Wide right counter-clockwise
        - l: Wide left counter-clockwise
        - u: Wide up counter-clockwise
        - d: Wide down counter-clockwise
        - f: Wide front counter-clockwise
        - b: Wide back counter-clockwise
        
        Cube Rotations:
        - x: Rotate entire cube counter-clockwise around x-axis
        - y: Rotate entire cube counter-clockwise around y-axis
        - z: Rotate entire cube counter-clockwise around z-axis

        Args:
            move (str): The move to perform (may include ' modifier, which will be stripped).
        """
        move_char = move.rstrip("'")
        
        if move_char == "U":
            move_slice = self.grid[:, 2, :].copy()
            for x in range(3):
                for z in range(3):
                    self.grid[z, 2, 2-x] = move_slice[x, z]
            for cubie in self.grid[:, 2, :].flatten():
                cubie.turn("U'")
                
        elif move_char == "D":
            move_slice = self.grid[:, 0, :].copy()
            for x in range(3):
                for z in range(3):
                    self.grid[2-z, 0, x] = move_slice[x, z]
            for cubie in self.grid[:, 0, :].flatten():
                cubie.turn("D'")
        
        elif move_char == "R":
            move_slice = self.grid[2, :, :].copy()
            for y in range(3):
                for z in range(3):
                    self.grid[2, 2-z, y] = move_slice[y, z]
            for cubie in self.grid[2, :, :].flatten():
                cubie.turn("R'")
        
        elif move_char == "L":
            move_slice = self.grid[0, :, :].copy()
            for y in range(3):
                for z in range(3):
                    self.grid[0, z, 2-y] = move_slice[y, z]
            for cubie in self.grid[0, :, :].flatten():
                cubie.turn("L'")

        elif move_char == "F":
            move_slice = self.grid[:, :, 2].copy()
            for x in range(3):
                for y in range(3):
                    self.grid[2-y, x, 2] = move_slice[x, y]
            for cubie in self.grid[:, :, 2].flatten():
                cubie.turn("F'")

        elif move_char == "B":
            move_slice = self.grid[:, :, 0].copy()
            for x in range(3):
                for y in range(3):
                    self.grid[y, 2-x, 0] = move_slice[x, y]
            for cubie in self.grid[:, :, 0].flatten():
                cubie.turn("B'")
        
        elif move_char == "M":
            move_slice = self.grid[1, :, :].copy()
            for y in range(3):
                for z in range(3):
                    self.grid[1, z, 2-y] = move_slice[y, z]
            for cubie in self.grid[1, :, :].flatten():
                if cubie:
                    cubie.turn("M'")
        
        elif move_char == "E":
            move_slice = self.grid[:, 1, :].copy()
            for x in range(3):
                for z in range(3):
                    self.grid[2-z, 1, x] = move_slice[x, z]
            for cubie in self.grid[:, 1, :].flatten():
                if cubie:
                    cubie.turn("E'")
        
        elif move_char == "S":
            move_slice = self.grid[:, :, 1].copy()
            for x in range(3):
                for y in range(3):
                    self.grid[2-y, x, 1] = move_slice[x, y]
            for cubie in self.grid[:, :, 1].flatten():
                if cubie:
                    cubie.turn("S'")
        
        elif move_char == "r":
            move_slices = self.grid[1:3, :, :].copy()
            for x_offset in range(2):
                x_coord = 1 + x_offset
                for y in range(3):
                    for z in range(3):
                        self.grid[x_coord, 2-z, y] = move_slices[x_offset, y, z]
            for cubie in self.grid[1, :, :].flatten():
                if cubie:
                    cubie.turn("R'")
            for cubie in self.grid[2, :, :].flatten():
                cubie.turn("R'")
        
        elif move_char == "l":
            move_slices = self.grid[0:2, :, :].copy()
            for x_offset in range(2):
                x_coord = x_offset
                for y in range(3):
                    for z in range(3):
                        self.grid[x_coord, z, 2-y] = move_slices[x_offset, y, z]
            for cubie in self.grid[0, :, :].flatten():
                cubie.turn("L'")
            for cubie in self.grid[1, :, :].flatten():
                if cubie:
                    cubie.turn("L'")
        
        elif move_char == "u":
            move_slices = self.grid[:, 1:3, :].copy()
            for y_offset in range(2):
                y_coord = 1 + y_offset
                for x in range(3):
                    for z in range(3):
                        self.grid[z, y_coord, 2-x] = move_slices[x, y_offset, z]
            for cubie in self.grid[:, 1, :].flatten():
                if cubie:
                    cubie.turn("U'")
            for cubie in self.grid[:, 2, :].flatten():
                cubie.turn("U'")
        
        elif move_char == "d":
            move_slices = self.grid[:, 0:2, :].copy()
            for y_offset in range(2):
                y_coord = y_offset
                for x in range(3):
                    for z in range(3):
                        self.grid[2-z, y_coord, x] = move_slices[x, y_offset, z]
            for cubie in self.grid[:, 0, :].flatten():
                cubie.turn("D'")
            for cubie in self.grid[:, 1, :].flatten():
                if cubie:
                    cubie.turn("D'")
        
        elif move_char == "f":
            move_slices = self.grid[:, :, 1:3].copy()
            for z_offset in range(2):
                z_coord = 1 + z_offset
                for x in range(3):
                    for y in range(3):
                        self.grid[2-y, x, z_coord] = move_slices[x, y, z_offset]
            for cubie in self.grid[:, :, 1].flatten():
                if cubie:
                    cubie.turn("F'")
            for cubie in self.grid[:, :, 2].flatten():
                cubie.turn("F'")
        
        elif move_char == "b":
            move_slices = self.grid[:, :, 0:2].copy()
            for z_offset in range(2):
                z_coord = z_offset
                for x in range(3):
                    for y in range(3):
                        self.grid[y, 2-x, z_coord] = move_slices[x, y, z_offset]
            for cubie in self.grid[:, :, 0].flatten():
                cubie.turn("B'")
            for cubie in self.grid[:, :, 1].flatten():
                if cubie:
                    cubie.turn("B'")
        
        elif move_char == "x":
            move_slices = self.grid[:, :, :].copy()
            for x in range(3):
                for y in range(3):
                    for z in range(3):
                        self.grid[x, 2-z, y] = move_slices[x, y, z]
            for cubie in self.grid.flatten():
                if cubie:
                    cubie.turn("X'")
        
        elif move_char == "y":
            move_slices = self.grid[:, :, :].copy()
            for x in range(3):
                for y in range(3):
                    for z in range(3):
                        self.grid[z, y, 2-x] = move_slices[x, y, z]
            for cubie in self.grid.flatten():
                if cubie:
                    cubie.turn("Y'")
        
        elif move_char == "z":
            move_slices = self.grid[:, :, :].copy()
            for x in range(3):
                for y in range(3):
                    for z in range(3):
                        self.grid[2-y, x, z] = move_slices[x, y, z]
            for cubie in self.grid.flatten():
                if cubie:
                    cubie.turn("Z'")
        
        else:
            raise ValueError(f"Invalid face '{move_char}' in counter-clockwise turn.")

    def _turn_double(self, move: str):
        """
        Turns a face or performs a cube rotation 180 degrees (double turn).
        
        This method performs a 180-degree rotation, which is equivalent to
        performing the same clockwise move twice, but is implemented as a
        single optimized transformation for better performance.
        
        The available moves are:
        
        Basic Face Moves:
        - U: Up face 180 degrees
        - D: Down face 180 degrees
        - L: Left face 180 degrees
        - R: Right face 180 degrees
        - F: Front face 180 degrees
        - B: Back face 180 degrees
        
        Slice Moves:
        - M: Middle slice 180 degrees
        - E: Equatorial slice 180 degrees
        - S: Standing slice 180 degrees
        
        Wide Moves:
        - r: Wide right 180 degrees
        - l: Wide left 180 degrees
        - u: Wide up 180 degrees
        - d: Wide down 180 degrees
        - f: Wide front 180 degrees
        - b: Wide back 180 degrees
        
        Cube Rotations:
        - x: Rotate entire cube 180 degrees around x-axis
        - y: Rotate entire cube 180 degrees around y-axis
        - z: Rotate entire cube 180 degrees around z-axis
        
        Note: A 180-degree turn is its own inverse (R2 followed by R2 returns
        to the original state).

        Args:
            move (str): The move to perform (single character, without '2' modifier).
        """
        if move == "U":
            move_slice = self.grid[:, 2, :].copy()
            for x in range(3):
                for z in range(3):
                    self.grid[2-x, 2, 2-z] = move_slice[x, z]
            for cubie in self.grid[:, 2, :].flatten():
                cubie.turn("U2")
                
        elif move == "D":
            move_slice = self.grid[:, 0, :].copy()
            for x in range(3):
                for z in range(3):
                    self.grid[2-x, 0, 2-z] = move_slice[x, z]
            for cubie in self.grid[:, 0, :].flatten():
                cubie.turn("D2")
        
        elif move == "R":
            move_slice = self.grid[2, :, :].copy()
            for y in range(3):
                for z in range(3):
                    self.grid[2, 2-y, 2-z] = move_slice[y, z]
            for cubie in self.grid[2, :, :].flatten():
                cubie.turn("R2")
        
        elif move == "L":
            move_slice = self.grid[0, :, :].copy()
            for y in range(3):
                for z in range(3):
                    self.grid[0, 2-y, 2-z] = move_slice[y, z]
            for cubie in self.grid[0, :, :].flatten():
                cubie.turn("L2")

        elif move == "F":
            move_slice = self.grid[:, :, 2].copy()
            for x in range(3):
                for y in range(3):
                    self.grid[2-x, 2-y, 2] = move_slice[x, y]
            for cubie in self.grid[:, :, 2].flatten():
                cubie.turn("F2")

        elif move == "B":
            move_slice = self.grid[:, :, 0].copy()
            for x in range(3):
                for y in range(3):
                    self.grid[2-x, 2-y, 0] = move_slice[x, y]
            for cubie in self.grid[:, :, 0].flatten():
                cubie.turn("B2")
        
        elif move == "M":
            move_slice = self.grid[1, :, :].copy()
            for y in range(3):
                for z in range(3):
                    self.grid[1, 2-y, 2-z] = move_slice[y, z]
            for cubie in self.grid[1, :, :].flatten():
                if cubie:
                    cubie.turn("M2")
        
        elif move == "E":
            move_slice = self.grid[:, 1, :].copy()
            for x in range(3):
                for z in range(3):
                    self.grid[2-x, 1, 2-z] = move_slice[x, z]
            for cubie in self.grid[:, 1, :].flatten():
                if cubie:
                    cubie.turn("E2")
        
        elif move == "S":
            move_slice = self.grid[:, :, 1].copy()
            for x in range(3):
                for y in range(3):
                    self.grid[2-x, 2-y, 1] = move_slice[x, y]
            for cubie in self.grid[:, :, 1].flatten():
                if cubie:
                    cubie.turn("S2")
        
        elif move == "r":
            move_slices = self.grid[1:3, :, :].copy()
            for x_offset in range(2):
                x_coord = 1 + x_offset
                for y in range(3):
                    for z in range(3):
                        self.grid[x_coord, 2-y, 2-z] = move_slices[x_offset, y, z]
            for cubie in self.grid[1, :, :].flatten():
                if cubie:
                    cubie.turn("R2")
            for cubie in self.grid[2, :, :].flatten():
                cubie.turn("R2")
        
        elif move == "l":
            move_slices = self.grid[0:2, :, :].copy()
            for x_offset in range(2):
                x_coord = x_offset
                for y in range(3):
                    for z in range(3):
                        self.grid[x_coord, 2-y, 2-z] = move_slices[x_offset, y, z]
            for cubie in self.grid[0, :, :].flatten():
                cubie.turn("L2")
            for cubie in self.grid[1, :, :].flatten():
                if cubie:
                    cubie.turn("L2")
        
        elif move == "u":
            move_slices = self.grid[:, 1:3, :].copy()
            for y_offset in range(2):
                y_coord = 1 + y_offset
                for x in range(3):
                    for z in range(3):
                        self.grid[2-x, y_coord, 2-z] = move_slices[x, y_offset, z]
            for cubie in self.grid[:, 1, :].flatten():
                if cubie:
                    cubie.turn("U2")
            for cubie in self.grid[:, 2, :].flatten():
                cubie.turn("U2")
        
        elif move == "d":
            move_slices = self.grid[:, 0:2, :].copy()
            for y_offset in range(2):
                y_coord = y_offset
                for x in range(3):
                    for z in range(3):
                        self.grid[2-x, y_coord, 2-z] = move_slices[x, y_offset, z]
            for cubie in self.grid[:, 0, :].flatten():
                cubie.turn("D2")
            for cubie in self.grid[:, 1, :].flatten():
                if cubie:
                    cubie.turn("D2")
        
        elif move == "f":
            move_slices = self.grid[:, :, 1:3].copy()
            for z_offset in range(2):
                z_coord = 1 + z_offset
                for x in range(3):
                    for y in range(3):
                        self.grid[2-x, 2-y, z_coord] = move_slices[x, y, z_offset]
            for cubie in self.grid[:, :, 1].flatten():
                if cubie:
                    cubie.turn("F2")
            for cubie in self.grid[:, :, 2].flatten():
                cubie.turn("F2")
        
        elif move == "b":
            move_slices = self.grid[:, :, 0:2].copy()
            for z_offset in range(2):
                z_coord = z_offset
                for x in range(3):
                    for y in range(3):
                        self.grid[2-x, 2-y, z_coord] = move_slices[x, y, z_offset]
            for cubie in self.grid[:, :, 0].flatten():
                cubie.turn("B2")
            for cubie in self.grid[:, :, 1].flatten():
                if cubie:
                    cubie.turn("B2")
        
        elif move == "x":
            move_slices = self.grid[:, :, :].copy()
            for x in range(3):
                for y in range(3):
                    for z in range(3):
                        self.grid[x, 2-y, 2-z] = move_slices[x, y, z]
            for cubie in self.grid.flatten():
                if cubie:
                    cubie.turn("X2")
        
        elif move == "y":
            move_slices = self.grid[:, :, :].copy()
            for x in range(3):
                for y in range(3):
                    for z in range(3):
                        self.grid[2-x, y, 2-z] = move_slices[x, y, z]
            for cubie in self.grid.flatten():
                if cubie:
                    cubie.turn("Y2")
        
        elif move == "z":
            move_slices = self.grid[:, :, :].copy()
            for x in range(3):
                for y in range(3):
                    for z in range(3):
                        self.grid[2-x, 2-y, z] = move_slices[x, y, z]
            for cubie in self.grid.flatten():
                if cubie:
                    cubie.turn("Z2")
        
        else:
            raise ValueError(f"Invalid face '{move}' in double turn.")


if __name__ == "__main__":
    my_cube = Cube()
    print("Initial cube state:")
    my_cube.show()
    
    print("\nPerforming scramble: U L2 D' L2 F R U D' F' L' B2 R D2 B2 R2 U2 F2 R' F2 R2 F2")
    my_cube.turn("U L2 D' L2 F R U D' F' L' B2 R D2 B2 R2 U2 F2 R' F2 R2 F2")
    my_cube.show()
    
    print("\nFor comprehensive testing, run: python tests/test_cuber.py")