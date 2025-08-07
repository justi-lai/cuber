class Cubie:
    """
    A class representing a single cubie on a Rubik's Cube.
    A cubie is a single piece of the cube, and can be either an edge, a corner, or a middle piece.
    """
    FACES = {"U", "D", "F", "B", "L", "R"}
    COLORS = {"W", "Y", "G", "B", "R", "O"}
    OPPOSITE_PAIRS = [("U", "D"), ("F", "B"), ("L", "R"), ("W", "Y"), ("G", "B"), ("R", "O")]

    _CLOCKWISE_MAPS = {
        "U": {"F": "L", "L": "B", "B": "R", "R": "F"},
        "D": {"B": "L", "L": "F", "F": "R", "R": "B"},
        "F": {"U": "R", "R": "D", "D": "L", "L": "U"},
        "B": {"D": "R", "R": "U", "U": "L", "L": "D"},
        "L": {"U": "F", "F": "D", "D": "B", "B": "U"},
        "R": {"D": "F", "F": "U", "U": "B", "B": "D"},
        "M": {"U": "F", "F": "D", "D": "B", "B": "U"},
        "E": {"B": "L", "L": "F", "F": "R", "R": "B"},
        "S": {"U": "R", "R": "D", "D": "L", "L": "U"},
        "X": {"D": "F", "F": "U", "U": "B", "B": "D"},
        "Y": {"F": "L", "L": "B", "B": "R", "R": "F"},
        "Z": {"U": "R", "R": "D", "D": "L", "L": "U"},
    }

    TRANSFORM_MAPS = {}
    for face, c_map in _CLOCKWISE_MAPS.items():
        TRANSFORM_MAPS[face] = c_map
        TRANSFORM_MAPS[f"{face}'"] = {v: k for k, v in c_map.items()}
        double_map = {}
        for start, end in c_map.items():
            double_map[start] = c_map[end]
        TRANSFORM_MAPS[f"{face}2"] = double_map

    def __init__(self, orientation: dict[str, str]):
        """
        Initialize a cubie with a dictionary of orientations.
        The orientation dictionary is a dictionary of the current faces of the cubie and the colors of the faces.
        The keys are the faces of the cubie, and the values are the colors of the faces.

        Args:
            orientation (dict[str, str]): The orientation of the cubie. The keys are the faces of the cubie, and the values are the colors of the faces.
        """
        self._validate_orientation(orientation)
        self.orientation = orientation
    
    def _validate_orientation(self, orientation: dict[str, str]):
        """
        Check if the orientation dictionary is valid.
        The orientation dictionary is a dictionary of the current faces of the cubie and the colors of the faces.
        The keys are the faces of the cubie, and the values are the colors of the faces.

        Args:
            orientation (dict[str, str]): The orientation of the cubie.

        Raises:
            ValueError: If the orientation dictionary is invalid.
        """
        faces = set(orientation.keys())
        colors = set(orientation.values())

        if not faces.issubset(self.FACES) or not colors.issubset(self.COLORS):
            raise ValueError("Invalid face or color in orientation")

        if len(faces) != len(colors):
            raise ValueError("Duplicate colors are not allowed on a single piece")

        for p1, p2 in self.OPPOSITE_PAIRS:
            if p1 in faces and p2 in faces:
                raise ValueError(f"Cannot have opposite faces {p1}/{p2} on one piece")
            if p1 in colors and p2 in colors:
                raise ValueError(f"Cannot have opposite colors {p1}/{p2} on one piece")
    
    def turn(self, move: str):
        """
        Updates the cubie's orientation based on a cube move.
        
        This method applies the transformation that would occur to this cubie
        when the specified move is performed on the cube. The cubie's orientation
        is updated to reflect how its faces would be repositioned.
        
        Supported moves include:
        - Basic face moves: U, D, L, R, F, B (and their primes/doubles)
        - Slice moves: M, E, S (and their primes/doubles)
        - Cube rotations: X, Y, Z (and their primes/doubles)
        
        The move can include modifiers:
        - ' (prime): Counter-clockwise (e.g., "U'", "R'", "X'")
        - 2: Double turn (e.g., "U2", "R2", "X2")
        
        Examples:
        - "R": Rotate this cubie as if the right face were turned clockwise
        - "U'": Rotate this cubie as if the up face were turned counter-clockwise
        - "M2": Rotate this cubie as if the middle slice were turned 180 degrees

        Args:
            move (str): The move to perform (e.g., "R", "U'", "M2", "X").

        Raises:
            ValueError: If the move is invalid or not recognized.
        """
        transform_map = self.TRANSFORM_MAPS.get(move.upper())
        if transform_map:
            self._apply_transform(transform_map)
        else:
            raise ValueError(f"Invalid move: {move}")

    def _apply_transform(self, transform_map: dict[str, str]):
        """
        Safely applies a transformation to the cubie's orientation.

        Args:
            transform_map (dict[str, str]): The transformation to apply.
        """
        new_orientation = {}
        for old_face, color in self.orientation.items():
            new_face = transform_map.get(old_face, old_face)
            new_orientation[new_face] = color
        self.orientation = new_orientation