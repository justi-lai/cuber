class Cubie:
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
        """
        self._validate_orientation(orientation)
        self.orientation = orientation
    
    def _validate_orientation(self, orientation: dict[str, str]):
        """Check if the orientation dictionary is valid."""
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
        Updates the orientation dictionary based on the move.
        """
        transform_map = self.TRANSFORM_MAPS.get(move)
        if transform_map:
            self._apply_transform(transform_map)
        else:
            raise ValueError(f"Invalid move: {move}")

    def _apply_transform(self, transform_map: dict[str, str]):
        """
        Safely applies a transformation to the cubie's orientation.
        """
        new_orientation = {}
        for old_face, color in self.orientation.items():
            new_face = transform_map.get(old_face, old_face)
            new_orientation[new_face] = color
        self.orientation = new_orientation