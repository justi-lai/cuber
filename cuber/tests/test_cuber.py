import unittest
import sys
import os
import numpy as np

# Add the parent directory to the Python path so we can import cuber and cubie
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ..cuber import Cube
    from ..cubie import Cubie
except ImportError:
    from cuber import Cube
    from cubie import Cubie


class TestCubie(unittest.TestCase):
    """Test cases for the Cubie class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.corner_cubie = Cubie({"R": "R", "U": "W", "F": "G"})
        self.edge_cubie = Cubie({"R": "R", "F": "G"})
        self.center_cubie = Cubie({"R": "R"})
    
    def test_cubie_initialization(self):
        """Test that cubies are initialized correctly."""
        self.assertEqual(self.corner_cubie.orientation, {"R": "R", "U": "W", "F": "G"})
        self.assertEqual(self.edge_cubie.orientation, {"R": "R", "F": "G"})
        self.assertEqual(self.center_cubie.orientation, {"R": "R"})
    
    def test_invalid_cubie_initialization(self):
        """Test that invalid cubie orientations raise ValueError."""
        # Test opposite faces on same piece
        with self.assertRaises(ValueError):
            Cubie({"R": "R", "L": "O"})
        
        # Test opposite colors on same piece
        with self.assertRaises(ValueError):
            Cubie({"R": "R", "U": "O"})
        
        # Test duplicate colors
        with self.assertRaises(ValueError):
            Cubie({"R": "R", "U": "R"})
    
    def test_cubie_turn_basic_moves(self):
        """Test basic face turns on cubies."""
        # Test R turn on corner cubie
        # R turn mapping: {"D": "F", "F": "U", "U": "B", "B": "D"}
        corner = Cubie({"R": "R", "U": "W", "B": "B"})
        corner.turn("R")
        expected = {"R": "R", "B": "W", "D": "B"}  # U->B, B->D after R turn
        self.assertEqual(corner.orientation, expected)
    
    def test_cubie_turn_prime_moves(self):
        """Test prime (counter-clockwise) turns on cubies."""
        # R' turn mapping (reverse of R): {"F": "D", "U": "F", "B": "U", "D": "B"}
        corner = Cubie({"R": "R", "U": "W", "B": "B"})
        corner.turn("R'")
        expected = {"R": "R", "F": "W", "U": "B"}  # U->F, B->U after R' turn
        self.assertEqual(corner.orientation, expected)
    
    def test_cubie_turn_double_moves(self):
        """Test double (180 degree) turns on cubies."""
        corner = Cubie({"R": "R", "U": "W", "F": "G"})
        corner.turn("R2")
        expected = {"R": "R", "D": "W", "B": "G"}  # U->D, F->B after R2 turn
        self.assertEqual(corner.orientation, expected)
    
    def test_invalid_move(self):
        """Test that invalid moves raise ValueError."""
        with self.assertRaises(ValueError):
            self.corner_cubie.turn("Q")  # Invalid move
        
        with self.assertRaises(ValueError):
            self.corner_cubie.turn("INVALID")  # Invalid move


class TestCube(unittest.TestCase):
    """Test cases for the Cube class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.cube = Cube()
    
    def test_cube_initialization(self):
        """Test that the cube is initialized in a solved state."""
        # Check that center cubie is None
        self.assertIsNone(self.cube.grid[1, 1, 1])
        
        # Check that corners, edges, and centers are properly initialized
        # Check a corner cubie
        corner = self.cube.grid[0, 0, 0]
        self.assertIsInstance(corner, Cubie)
        expected_faces = {"L", "D", "B"}
        self.assertEqual(set(corner.orientation.keys()), expected_faces)
        
        # Check an edge cubie
        edge = self.cube.grid[1, 0, 0]
        self.assertIsInstance(edge, Cubie)
        expected_faces = {"D", "B"}
        self.assertEqual(set(edge.orientation.keys()), expected_faces)
        
        # Check a face center
        center = self.cube.grid[2, 1, 1]
        self.assertIsInstance(center, Cubie)
        expected_faces = {"R"}
        self.assertEqual(set(center.orientation.keys()), expected_faces)
    
    def test_basic_moves(self):
        """Test basic face moves."""
        # Test that moves don't crash
        self.cube.turn("R")
        self.cube.turn("U")
        self.cube.turn("F")
        self.cube.turn("L")
        self.cube.turn("D")
        self.cube.turn("B")
    
    def test_prime_moves(self):
        """Test prime (counter-clockwise) moves."""
        self.cube.turn("R'")
        self.cube.turn("U'")
        self.cube.turn("F'")
        self.cube.turn("L'")
        self.cube.turn("D'")
        self.cube.turn("B'")
    
    def test_double_moves(self):
        """Test double (180 degree) moves."""
        self.cube.turn("R2")
        self.cube.turn("U2")
        self.cube.turn("F2")
        self.cube.turn("L2")
        self.cube.turn("D2")
        self.cube.turn("B2")
    
    def test_slice_moves(self):
        """Test slice moves (M, E, S)."""
        self.cube.turn("M")
        self.cube.turn("E")
        self.cube.turn("S")
        self.cube.turn("M'")
        self.cube.turn("E'")
        self.cube.turn("S'")
        self.cube.turn("M2")
        self.cube.turn("E2")
        self.cube.turn("S2")
    
    def test_wide_moves(self):
        """Test wide moves (r, l, u, d, f, b)."""
        self.cube.turn("r")
        self.cube.turn("l")
        self.cube.turn("u")
        self.cube.turn("d")
        self.cube.turn("f")
        self.cube.turn("b")
        self.cube.turn("r'")
        self.cube.turn("l'")
        self.cube.turn("u'")
        self.cube.turn("d'")
        self.cube.turn("f'")
        self.cube.turn("b'")
    
    def test_get_face_map_solved_cube(self):
        """Test _get_face_map method on a solved cube."""
        # Test all faces on a solved cube
        expected_colors = {
            'U': 'W',  # Up face should be all white
            'D': 'Y',  # Down face should be all yellow
            'L': 'O',  # Left face should be all orange
            'R': 'R',  # Right face should be all red
            'F': 'G',  # Front face should be all green
            'B': 'B'   # Back face should be all blue
        }
        
        for face, expected_color in expected_colors.items():
            face_map = self.cube._get_face_map(face)
            
            # Check that it's a 3x3 array
            self.assertEqual(face_map.shape, (3, 3))
            
            # Check that all squares are the expected color
            for i in range(3):
                for j in range(3):
                    self.assertEqual(face_map[i, j], expected_color,
                                   f"Face {face} at position ({i},{j}) should be {expected_color}")
    
    def test_get_face_map_after_moves(self):
        """Test _get_face_map method after performing moves."""
        # Apply a simple R move and check that faces change appropriately
        initial_up = self.cube._get_face_map('U').copy()
        initial_right = self.cube._get_face_map('R').copy()
        
        self.cube.turn("R")
        
        after_up = self.cube._get_face_map('U')
        after_right = self.cube._get_face_map('R')
        
        # Right face should have rotated (not all squares same color anymore)
        # We can't predict exact colors due to complexity, but we can check structure
        self.assertEqual(after_right.shape, (3, 3))
        self.assertEqual(after_up.shape, (3, 3))
        
        # Apply R' to return to original state
        self.cube.turn("R'")
        restored_up = self.cube._get_face_map('U')
        restored_right = self.cube._get_face_map('R')
        
        # Should be back to original state
        np.testing.assert_array_equal(initial_up, restored_up)
        np.testing.assert_array_equal(initial_right, restored_right)
    
    def test_get_face_map_invalid_face(self):
        """Test _get_face_map method with invalid face character."""
        with self.assertRaises(KeyError):
            self.cube._get_face_map('X')  # Invalid face
        
        with self.assertRaises(KeyError):
            self.cube._get_face_map('Z')  # Invalid face
    
    def test_get_face_method(self):
        """Test the public get_face method."""
        # Test single face
        face_dict = self.cube.get_faces('U')
        self.assertIn('U', face_dict)
        self.assertEqual(len(face_dict['U']), 3)  # 3 rows
        self.assertEqual(len(face_dict['U'][0]), 3)  # 3 columns
        
        # All squares should be 'W' on solved cube's up face
        for row in face_dict['U']:
            for square in row:
                self.assertEqual(square, 'W')
        
        # Test multiple faces
        multiple_faces = self.cube.get_faces('UDL')
        self.assertEqual(len(multiple_faces), 3)
        self.assertIn('U', multiple_faces)
        self.assertIn('D', multiple_faces)
        self.assertIn('L', multiple_faces)
        
        # Check expected colors
        for row in multiple_faces['U']:
            for square in row:
                self.assertEqual(square, 'W')
        for row in multiple_faces['D']:
            for square in row:
                self.assertEqual(square, 'Y')
        for row in multiple_faces['L']:
            for square in row:
                self.assertEqual(square, 'O')
    
    def test_get_face_all_faces(self):
        """Test get_face method with all faces."""
        all_faces = self.cube.get_faces('UDLRFB')
        self.assertEqual(len(all_faces), 6)
        
        expected_colors = {'U': 'W', 'D': 'Y', 'L': 'O', 'R': 'R', 'F': 'G', 'B': 'B'}
        
        for face, expected_color in expected_colors.items():
            self.assertIn(face, all_faces)
            for row in all_faces[face]:
                for square in row:
                    self.assertEqual(square, expected_color)
    
    def test_get_faces_str_single_face(self):
        """Test get_faces_str method with a single face."""
        # Test with solved cube
        result = self.cube.get_faces_str('U')
        expected = "W W W W W W W W W"
        self.assertEqual(result, expected)
        
        # Test each face individually on solved cube
        expected_faces = {
            'U': "W W W W W W W W W",
            'D': "Y Y Y Y Y Y Y Y Y", 
            'L': "O O O O O O O O O",
            'R': "R R R R R R R R R",
            'F': "G G G G G G G G G",
            'B': "B B B B B B B B B"
        }
        
        for face, expected_str in expected_faces.items():
            with self.subTest(face=face):
                result = self.cube.get_faces_str(face)
                self.assertEqual(result, expected_str)
    
    def test_get_faces_str_multiple_faces(self):
        """Test get_faces_str method with multiple faces."""
        # Test two faces
        result = self.cube.get_faces_str('UD')
        expected = "W W W W W W W W W Y Y Y Y Y Y Y Y Y"
        self.assertEqual(result, expected)
        
        # Test three faces
        result = self.cube.get_faces_str('UDL')
        expected = "W W W W W W W W W Y Y Y Y Y Y Y Y Y O O O O O O O O O"
        self.assertEqual(result, expected)
        
        # Test all faces
        result = self.cube.get_faces_str('UDLRFB')
        expected = "W W W W W W W W W Y Y Y Y Y Y Y Y Y O O O O O O O O O R R R R R R R R R G G G G G G G G G B B B B B B B B B"
        self.assertEqual(result, expected)
    
    def test_get_faces_str_empty_input(self):
        """Test get_faces_str method with empty input."""
        result = self.cube.get_faces_str('')
        self.assertEqual(result, "")
    
    def test_get_faces_str_scrambled_cube(self):
        """Test get_faces_str method after scrambling the cube."""
        # Apply some moves to scramble
        self.cube.turn("R U R' U'")
        
        # Test that it returns a string (exact content will vary)
        result = self.cube.get_faces_str('U')
        self.assertIsInstance(result, str)
        
        # Should have 17 characters (9 colors + 8 spaces)
        self.assertEqual(len(result), 17)
        
        # Should contain only valid colors and spaces
        valid_chars = set('WYGROBR ')
        self.assertTrue(all(c in valid_chars for c in result))
    
    def test_get_faces_str_custom_stickers(self):
        """Test get_faces_str method with custom sticker initialization."""
        # Create cube with custom stickers
        custom_stickers = {
            'U': [['R', 'W', 'G'], ['B', 'W', 'Y'], ['O', 'W', 'R']],
            'D': [['Y', 'Y', 'Y'], ['Y', 'Y', 'Y'], ['Y', 'Y', 'Y']],
            'L': [['O', 'O', 'O'], ['O', 'O', 'O'], ['O', 'O', 'O']],
            'R': [['R', 'R', 'R'], ['R', 'R', 'R'], ['R', 'R', 'R']],
            'F': [['G', 'G', 'G'], ['G', 'G', 'G'], ['G', 'G', 'G']],
            'B': [['B', 'B', 'B'], ['B', 'B', 'B'], ['B', 'B', 'B']]
        }
        
        custom_cube = Cube(custom_stickers)
        result = custom_cube.get_faces_str('U')
        expected = "R W G B W Y O W R"
        self.assertEqual(result, expected)
    
    def test_get_faces_str_invalid_face(self):
        """Test get_faces_str method with invalid face character."""
        with self.assertRaises(ValueError):
            self.cube.get_faces_str('Z')
    
    def test_get_faces_str_face_order(self):
        """Test that get_faces_str respects the order of faces provided."""
        # Test different orderings give different results
        result1 = self.cube.get_faces_str('UD')
        result2 = self.cube.get_faces_str('DU')
        
        # They should be different (U colors first vs D colors first)
        self.assertNotEqual(result1, result2)
        
        # But both should be valid strings
        self.assertIsInstance(result1, str)
        self.assertIsInstance(result2, str)
        
        # Test a complex reordering
        faces_abcd = self.cube.get_faces_str('UDLR')
        faces_dcba = self.cube.get_faces_str('RLDU')
        self.assertNotEqual(faces_abcd, faces_dcba)

    def test_reset_method(self):
        """Test the reset method returns cube to solved state."""
        # Verify cube starts solved
        initial_state = self._get_cube_state()
        
        # Scramble the cube
        scramble = "R U R' U' F' U F U R U2 R' U2 R U' R'"
        self.cube.turn(scramble)
        
        # Verify cube is scrambled (not in solved state)
        scrambled_state = self._get_cube_state()
        self.assertNotEqual(initial_state, scrambled_state)
        
        # Reset the cube
        self.cube.reset()
        
        # Verify cube is back to solved state
        reset_state = self._get_cube_state()
        self._assert_states_equal(initial_state, reset_state)
    
    def test_reset_all_faces_solved(self):
        """Test that reset method results in all faces showing correct colors."""
        # Scramble the cube first
        self.cube.turn("R U R' U' F R F' U R U' R'")
        
        # Reset the cube
        self.cube.reset()
        
        # Check all faces are correctly colored
        expected_colors = {
            'U': 'W',  # Up face should be all white
            'D': 'Y',  # Down face should be all yellow
            'L': 'O',  # Left face should be all orange
            'R': 'R',  # Right face should be all red
            'F': 'G',  # Front face should be all green
            'B': 'B'   # Back face should be all blue
        }
        
        for face, expected_color in expected_colors.items():
            face_map = self.cube._get_face_map(face)
            for i in range(3):
                for j in range(3):
                    self.assertEqual(face_map[i, j], expected_color,
                                   f"After reset, face {face} at position ({i},{j}) should be {expected_color}")
    
    def test_reset_multiple_times(self):
        """Test that reset works consistently when called multiple times."""
        # Get initial solved state
        initial_state = self._get_cube_state()
        
        # Reset should not change a solved cube
        self.cube.reset()
        after_first_reset = self._get_cube_state()
        self._assert_states_equal(initial_state, after_first_reset)
        
        # Scramble and reset multiple times
        for i in range(3):
            # Scramble with different moves each time
            scrambles = ["R U R'", "F D F'", "L' U L U2"]
            self.cube.turn(scrambles[i])
            
            # Reset should always return to solved state
            self.cube.reset()
            reset_state = self._get_cube_state()
            self._assert_states_equal(initial_state, reset_state)
    
    def test_reset_preserves_cube_structure(self):
        """Test that reset preserves the cube's 3D grid structure."""
        # Scramble the cube
        self.cube.turn("R U R' U' F R F'")
        
        # Reset the cube
        self.cube.reset()
        
        # Verify grid structure is maintained
        self.assertEqual(self.cube.grid.shape, (3, 3, 3))
        
        # Verify center is still None
        self.assertIsNone(self.cube.grid[1, 1, 1])
        
        # Verify all other positions have Cubie objects
        for x, y, z in np.ndindex(3, 3, 3):
            if x == 1 and y == 1 and z == 1:
                continue
            self.assertIsInstance(self.cube.grid[x, y, z], Cubie)
            self.assertIsNotNone(self.cube.grid[x, y, z].orientation)
    
    def test_reset_vs_new_cube(self):
        """Test that reset produces the same state as creating a new cube."""
        # Scramble the current cube
        self.cube.turn("R U2 R' D R U' R' D' R U' R'")
        
        # Reset the cube
        self.cube.reset()
        reset_state = self._get_cube_state()
        
        # Create a new cube
        new_cube = Cube()
        new_state = self._get_cube_state_from_cube(new_cube)
        
        # Both should be identical
        self._assert_states_equal(reset_state, new_state)
    
    def test_is_solved_new_cube(self):
        """Test that a new cube is solved."""
        self.assertTrue(self.cube.is_solved())
    
    def test_is_solved_after_scramble(self):
        """Test that cube is not solved after scrambling."""
        self.assertTrue(self.cube.is_solved())  # Start solved
        
        # Scramble the cube
        self.cube.turn("R U R' U'")
        self.assertFalse(self.cube.is_solved())  # Should not be solved
        
        # More complex scramble
        self.cube.turn("F D F' U R U' R'")
        self.assertFalse(self.cube.is_solved())  # Still not solved
    
    def test_is_solved_after_reset(self):
        """Test that cube is solved after reset."""
        # Scramble the cube
        self.cube.turn("R U R' U' F' U F U R U2 R' U2 R U' R'")
        self.assertFalse(self.cube.is_solved())
        
        # Reset should make it solved
        self.cube.reset()
        self.assertTrue(self.cube.is_solved())
    
    def test_is_solved_with_cube_rotations(self):
        """Test that cube rotations don't affect solved status."""
        # Start with solved cube
        self.assertTrue(self.cube.is_solved())
        
        # Apply various cube rotations - should still be solved
        rotations = ["x", "y", "z", "x'", "y'", "z'", "x2", "y2", "z2"]
        for rotation in rotations:
            self.cube.turn(rotation)
            self.assertTrue(self.cube.is_solved(), 
                          f"Cube should still be solved after rotation {rotation}")
    
    def test_is_solved_complex_rotation_sequence(self):
        """Test solved status with complex cube rotation sequences."""
        # Start with solved cube
        self.assertTrue(self.cube.is_solved())
        
        # Apply complex rotation sequence
        self.cube.turn("x y z x' y' z' x2 y2 z2")
        self.assertTrue(self.cube.is_solved())
        
        # Another complex sequence
        self.cube.turn("x y x' z y' z' x y z")
        self.assertTrue(self.cube.is_solved())
    
    def test_is_solved_identity_sequences(self):
        """Test that identity sequences maintain solved status."""
        # Test that move + inverse keeps cube solved
        identity_sequences = [
            "R R'", "U U'", "F F'", "L L'", "D D'", "B B'",
            "R R R R", "U2 U2", "M M'", "E E'", "S S'",
            "r r'", "u u'", "f f'"
        ]
        
        for sequence in identity_sequences:
            self.cube.reset()  # Start fresh
            self.assertTrue(self.cube.is_solved())
            
            self.cube.turn(sequence)
            self.assertTrue(self.cube.is_solved(), 
                          f"Cube should remain solved after identity sequence: {sequence}")
    
    def test_is_solved_mixed_sequences(self):
        """Test solved status detection with mixed move sequences."""
        # Test a sequence that returns to solved state
        self.cube.turn("R U R' U' R U R' U'")  # Two sexy moves
        # This might or might not be solved depending on the pattern
        
        # Test definitely non-solving sequence
        self.cube.reset()
        self.cube.turn("R U R'")  # Incomplete pattern
        self.assertFalse(self.cube.is_solved())
        
        # Test sequence with cube rotations mixed in
        self.cube.reset()
        self.cube.turn("x R U R' x'")  # Rotation + moves + rotation back
        # This should not be solved as it's not an identity
        # (Note: this test checks the logic works with mixed move types)
    
    def test_reverse_formula_basic_moves(self):
        """Test reverse_formula with basic face moves."""
        test_cases = [
            ("R", "R'"),
            ("R'", "R"),
            ("R2", "R2"),
            ("U", "U'"),
            ("U'", "U"),
            ("U2", "U2"),
        ]
        
        for original, expected in test_cases:
            result = Cube.reverse_formula(original)
            self.assertEqual(result, expected, 
                           f"reverse_formula('{original}') should be '{expected}', got '{result}'")
    
    def test_reverse_formula_sequences(self):
        """Test reverse_formula with move sequences."""
        test_cases = [
            ("R U", "U' R'"),
            ("R U R'", "R U' R'"),
            ("R U R' U'", "U R U' R'"),
            ("R2 U' F D2", "D2 F' U R2"),
            ("L D L' D'", "D L D' L'"),
        ]
        
        for original, expected in test_cases:
            result = Cube.reverse_formula(original)
            self.assertEqual(result, expected,
                           f"reverse_formula('{original}') should be '{expected}', got '{result}'")
    
    def test_reverse_formula_all_move_types(self):
        """Test reverse_formula with different move types."""
        # Test all basic moves
        basic_moves = "U D L R F B"
        result = Cube.reverse_formula(basic_moves)
        expected = "B' F' R' L' D' U'"
        self.assertEqual(result, expected)
        
        # Test slice moves
        slice_moves = "M E S"
        result = Cube.reverse_formula(slice_moves)
        expected = "S' E' M'"
        self.assertEqual(result, expected)
        
        # Test wide moves
        wide_moves = "r l u d f b"
        result = Cube.reverse_formula(wide_moves)
        expected = "b' f' d' u' l' r'"
        self.assertEqual(result, expected)
        
        # Test cube rotations
        rotations = "x y z"
        result = Cube.reverse_formula(rotations)
        expected = "z' y' x'"
        self.assertEqual(result, expected)
    
    def test_reverse_formula_mixed_modifiers(self):
        """Test reverse_formula with mixed prime and double moves."""
        test_cases = [
            ("R U' R2 F' D", "D' F R2 U R'"),
            ("x' y2 z M' E2 S", "S' E2 M z' y2 x"),  # Fixed: z becomes z'
            ("r' u2 f d' l2 b", "b' l2 d f' u2 r"),
        ]
        
        for original, expected in test_cases:
            result = Cube.reverse_formula(original)
            self.assertEqual(result, expected,
                           f"reverse_formula('{original}') should be '{expected}', got '{result}'")
    
    def test_reverse_formula_identity_property(self):
        """Test that applying formula + reversed formula returns to solved state."""
        test_formulas = [
            "R U R' U'",
            "R U R' F' R U R' U' R' F R2 U' R'",  # T-Perm
            "M2 U M2 U2 M2 U M2",  # H-Perm
            "x y z x' y' z'",  # Rotation sequence
            "r U r' F R F'",  # Wide move sequence
        ]
        
        for formula in test_formulas:
            # Start with solved cube
            self.cube.reset()
            self.assertTrue(self.cube.is_solved())
            
            # Apply formula
            self.cube.turn(formula)
            
            # Apply reversed formula
            reversed_formula = Cube.reverse_formula(formula)
            self.cube.turn(reversed_formula)
            
            # Should be back to solved state
            self.assertTrue(self.cube.is_solved(),
                          f"Formula '{formula}' + reversed should return to solved state")
    
    def test_reverse_formula_empty_string(self):
        """Test reverse_formula with empty string."""
        result = Cube.reverse_formula("")
        self.assertEqual(result, "")
    
    def test_reverse_formula_single_space(self):
        """Test reverse_formula with strings containing extra spaces."""
        # Should handle extra spaces gracefully
        result = Cube.reverse_formula("R  U   R'")  # Extra spaces
        expected = "R U' R'"
        self.assertEqual(result, expected)
    
    def test_reverse_formula_complex_algorithms(self):
        """Test reverse_formula with complex algorithms."""
        # Test with some well-known algorithms
        algorithms = [
            ("R U R' U' R U R' U' R U R' U' R U R' U'", # Sexy move x4
             "U R U' R' U R U' R' U R U' R' U R U' R'"),
            ("R U2 R' D R U' R' D'",  # A basic commutator
             "D R U R' D' R U2 R'"),
        ]
        
        for original, expected in algorithms:
            result = Cube.reverse_formula(original)
            self.assertEqual(result, expected)
            
            # Also test that it actually works on the cube
            self.cube.reset()
            self.cube.turn(original)
            self.cube.turn(result)
            self.assertTrue(self.cube.is_solved())
    
    def test_cube_rotations(self):
        """Test cube rotations (x, y, z)."""
        self.cube.turn("x")
        self.cube.turn("y")
        self.cube.turn("z")
        self.cube.turn("x'")
        self.cube.turn("y'")
        self.cube.turn("z'")
        self.cube.turn("x2")
        self.cube.turn("y2")
        self.cube.turn("z2")
    
    def test_move_sequence(self):
        """Test a sequence of moves."""
        moves = "R U R' U' R U R' U'"
        self.cube.turn(moves)
        # Should not crash
    
    def test_move_cancellation(self):
        """Test that a move followed by its inverse returns to original state."""
        # Save initial state
        initial_state = self._get_cube_state()
        
        # Apply R then R'
        self.cube.turn("R")
        self.cube.turn("R'")
        
        # Should be back to initial state
        final_state = self._get_cube_state()
        self._assert_states_equal(initial_state, final_state)
    
    def test_double_move_equivalence(self):
        """Test that a double move is equivalent to doing the move twice."""
        # Save initial state
        initial_state = self._get_cube_state()
        
        # Apply R2
        cube1 = Cube()
        cube1.turn("R2")
        state1 = self._get_cube_state_from_cube(cube1)
        
        # Apply R R
        cube2 = Cube()
        cube2.turn("R R")
        state2 = self._get_cube_state_from_cube(cube2)
        
        # Should be the same
        self._assert_states_equal(state1, state2)
    
    def test_four_quarter_turns_identity(self):
        """Test that four quarter turns return to original state."""
        initial_state = self._get_cube_state()
        
        # Apply R four times
        self.cube.turn("R R R R")
        
        final_state = self._get_cube_state()
        self._assert_states_equal(initial_state, final_state)
    
    def test_invalid_move(self):
        """Test that invalid moves raise ValueError."""
        with self.assertRaises(ValueError):
            self.cube.turn("X")  # Should be lowercase x
        
        with self.assertRaises(ValueError):
            self.cube.turn("Q")  # Invalid move
    
    def test_empty_move_string(self):
        """Test that empty move strings are handled gracefully."""
        initial_state = self._get_cube_state()
        self.cube.turn("")
        final_state = self._get_cube_state()
        self._assert_states_equal(initial_state, final_state)
    
    def test_multiple_spaces_in_moves(self):
        """Test that multiple spaces in move strings are handled correctly."""
        self.cube.turn("R  U   R'    U'")  # Should not crash
    
    def _get_cube_state(self):
        """Helper method to get the current state of self.cube."""
        return self._get_cube_state_from_cube(self.cube)
    
    def _get_cube_state_from_cube(self, cube):
        """Helper method to get the state of a given cube."""
        state = {}
        for x, y, z in np.ndindex(3, 3, 3):
            if cube.grid[x, y, z] is not None:
                state[(x, y, z)] = cube.grid[x, y, z].orientation.copy()
        return state
    
    def _assert_states_equal(self, state1, state2):
        """Helper method to assert that two cube states are equal."""
        self.assertEqual(set(state1.keys()), set(state2.keys()))
        for pos in state1:
            self.assertEqual(state1[pos], state2[pos], 
                           f"States differ at position {pos}")

    def test_sticker_initialization_dict_solved(self):
        """Test sticker initialization with dictionary format - solved cube."""
        solved_dict = {
            'U': [['W', 'W', 'W'], ['W', 'W', 'W'], ['W', 'W', 'W']],
            'D': [['Y', 'Y', 'Y'], ['Y', 'Y', 'Y'], ['Y', 'Y', 'Y']],
            'L': [['O', 'O', 'O'], ['O', 'O', 'O'], ['O', 'O', 'O']],
            'R': [['R', 'R', 'R'], ['R', 'R', 'R'], ['R', 'R', 'R']],
            'F': [['G', 'G', 'G'], ['G', 'G', 'G'], ['G', 'G', 'G']],
            'B': [['B', 'B', 'B'], ['B', 'B', 'B'], ['B', 'B', 'B']]
        }
        
        cube = Cube(solved_dict)
        self.assertTrue(cube.is_solved())
        self.assertTrue(cube.is_valid())
        
        # Verify each face matches expected colors
        expected_colors = {'U': 'W', 'D': 'Y', 'L': 'O', 'R': 'R', 'F': 'G', 'B': 'B'}
        for face, expected_color in expected_colors.items():
            face_map = cube._get_face_map(face)
            for i in range(3):
                for j in range(3):
                    self.assertEqual(face_map[i, j], expected_color,
                                   f"Face {face} at position ({i},{j}) should be {expected_color}")

    def test_sticker_initialization_dict_scrambled(self):
        """Test sticker initialization with dictionary format - scrambled cube."""
        scrambled_dict = {
            'U': [['W', 'R', 'W'], ['G', 'W', 'B'], ['W', 'Y', 'W']],
            'D': [['Y', 'O', 'Y'], ['R', 'Y', 'G'], ['Y', 'W', 'Y']],
            'L': [['O', 'W', 'O'], ['B', 'O', 'Y'], ['O', 'R', 'O']],
            'R': [['R', 'G', 'R'], ['W', 'R', 'O'], ['R', 'B', 'R']],
            'F': [['G', 'Y', 'G'], ['O', 'G', 'R'], ['G', 'W', 'G']],
            'B': [['B', 'R', 'B'], ['Y', 'B', 'W'], ['B', 'G', 'B']]
        }
        
        cube = Cube(scrambled_dict)
        self.assertFalse(cube.is_solved())
        self.assertFalse(cube.is_valid())  # This particular scramble has wrong color counts
        
        # Verify specific positions match input
        u_face = cube._get_face_map('U')
        self.assertEqual(u_face[0, 1], 'R')  # Top middle should be red
        self.assertEqual(u_face[1, 1], 'W')  # Center should be white
        self.assertEqual(u_face[2, 1], 'Y')  # Bottom middle should be yellow

    def test_sticker_initialization_list_solved(self):
        """Test sticker initialization with 1D list format - solved cube."""
        # Order: U, L, F, R, B, D (9 stickers each)
        solved_list = ['W']*9 + ['O']*9 + ['G']*9 + ['R']*9 + ['B']*9 + ['Y']*9
        
        cube = Cube(solved_list)
        self.assertTrue(cube.is_solved())
        self.assertTrue(cube.is_valid())
        
        # Verify each face has correct color
        expected_colors = {'U': 'W', 'D': 'Y', 'L': 'O', 'R': 'R', 'F': 'G', 'B': 'B'}
        for face, expected_color in expected_colors.items():
            face_map = cube._get_face_map(face)
            for i in range(3):
                for j in range(3):
                    self.assertEqual(face_map[i, j], expected_color)

    def test_sticker_initialization_list_custom(self):
        """Test sticker initialization with 1D list format - custom pattern."""
        # Create a specific pattern: U face with different colors in specific positions
        u_pattern = ['W', 'R', 'W', 'G', 'W', 'B', 'W', 'Y', 'W']  # W with colored edges
        custom_list = u_pattern + ['O']*9 + ['G']*9 + ['R']*9 + ['B']*9 + ['Y']*9
        
        cube = Cube(custom_list)
        
        # Verify the U face pattern
        u_face = cube._get_face_map('U')
        expected_u = np.array([['W', 'R', 'W'], ['G', 'W', 'B'], ['W', 'Y', 'W']])
        np.testing.assert_array_equal(u_face, expected_u)

    def test_sticker_initialization_invalid_dict_missing_face(self):
        """Test that dictionary initialization fails with missing faces."""
        incomplete_dict = {
            'U': [['W', 'W', 'W'], ['W', 'W', 'W'], ['W', 'W', 'W']],
            'D': [['Y', 'Y', 'Y'], ['Y', 'Y', 'Y'], ['Y', 'Y', 'Y']],
            'L': [['O', 'O', 'O'], ['O', 'O', 'O'], ['O', 'O', 'O']],
            'R': [['R', 'R', 'R'], ['R', 'R', 'R'], ['R', 'R', 'R']],
            'F': [['G', 'G', 'G'], ['G', 'G', 'G'], ['G', 'G', 'G']]
            # Missing 'B' face
        }
        
        with self.assertRaises(ValueError) as context:
            Cube(incomplete_dict)
        self.assertIn("exactly these face keys", str(context.exception))

    def test_sticker_initialization_invalid_dict_wrong_size(self):
        """Test that dictionary initialization fails with wrong face size."""
        invalid_size_dict = {
            'U': [['W', 'W'], ['W', 'W']],  # 2x2 instead of 3x3
            'D': [['Y', 'Y', 'Y'], ['Y', 'Y', 'Y'], ['Y', 'Y', 'Y']],
            'L': [['O', 'O', 'O'], ['O', 'O', 'O'], ['O', 'O', 'O']],
            'R': [['R', 'R', 'R'], ['R', 'R', 'R'], ['R', 'R', 'R']],
            'F': [['G', 'G', 'G'], ['G', 'G', 'G'], ['G', 'G', 'G']],
            'B': [['B', 'B', 'B'], ['B', 'B', 'B'], ['B', 'B', 'B']]
        }
        
        with self.assertRaises(ValueError) as context:
            Cube(invalid_size_dict)
        self.assertIn("must be a 3x3 array", str(context.exception))

    def test_sticker_initialization_invalid_list_wrong_length(self):
        """Test that list initialization fails with wrong length."""
        with self.assertRaises(ValueError) as context:
            Cube(['W'] * 50)  # Should be 54
        self.assertIn("exactly 54 colors", str(context.exception))
        
        with self.assertRaises(ValueError) as context:
            Cube(['W'] * 60)  # Should be 54
        self.assertIn("exactly 54 colors", str(context.exception))

    def test_sticker_initialization_invalid_type(self):
        """Test that initialization fails with invalid type."""
        with self.assertRaises(ValueError) as context:
            Cube("invalid_string")
        self.assertIn("must be None, dict, or list", str(context.exception))
        
        with self.assertRaises(ValueError) as context:
            Cube(123)
        self.assertIn("must be None, dict, or list", str(context.exception))

    def test_is_valid_solved_cube(self):
        """Test that is_valid returns True for a solved cube."""
        cube = Cube()
        self.assertTrue(cube.is_valid())

    def test_is_valid_after_moves(self):
        """Test that is_valid returns True after valid moves."""
        cube = Cube()
        cube.turn("R U R' U'")  # Sexy move
        self.assertTrue(cube.is_valid())

    def test_is_valid_invalid_color_count(self):
        """Test that is_valid returns False for invalid color counts."""
        invalid_dict = {
            'U': [['W', 'W', 'W'], ['W', 'W', 'W'], ['W', 'W', 'W']],
            'D': [['W', 'W', 'W'], ['W', 'W', 'W'], ['W', 'W', 'W']],  # Too many W's
            'L': [['O', 'O', 'O'], ['O', 'O', 'O'], ['O', 'O', 'O']],
            'R': [['R', 'R', 'R'], ['R', 'R', 'R'], ['R', 'R', 'R']],
            'F': [['G', 'G', 'G'], ['G', 'G', 'G'], ['G', 'G', 'G']],
            'B': [['B', 'B', 'B'], ['B', 'B', 'B'], ['B', 'B', 'B']]
        }
        
        cube = Cube(invalid_dict)
        self.assertFalse(cube.is_valid())

    def test_is_valid_duplicate_centers(self):
        """Test that is_valid returns False for duplicate center colors."""
        invalid_centers_dict = {
            'U': [['W', 'W', 'W'], ['W', 'W', 'W'], ['W', 'W', 'W']],
            'D': [['Y', 'Y', 'Y'], ['Y', 'W', 'Y'], ['Y', 'Y', 'Y']],  # W center (same as U)
            'L': [['O', 'O', 'O'], ['O', 'O', 'O'], ['O', 'O', 'O']],
            'R': [['R', 'R', 'R'], ['R', 'R', 'R'], ['R', 'R', 'R']],
            'F': [['G', 'G', 'G'], ['G', 'G', 'G'], ['G', 'G', 'G']],
            'B': [['B', 'B', 'B'], ['B', 'B', 'B'], ['B', 'B', 'B']]
        }
        
        cube = Cube(invalid_centers_dict)
        self.assertFalse(cube.is_valid())

    def test_sticker_dict_vs_solved_equivalence(self):
        """Test that dictionary initialization with solved pattern equals default cube."""
        solved_dict = {
            'U': [['W', 'W', 'W'], ['W', 'W', 'W'], ['W', 'W', 'W']],
            'D': [['Y', 'Y', 'Y'], ['Y', 'Y', 'Y'], ['Y', 'Y', 'Y']],
            'L': [['O', 'O', 'O'], ['O', 'O', 'O'], ['O', 'O', 'O']],
            'R': [['R', 'R', 'R'], ['R', 'R', 'R'], ['R', 'R', 'R']],
            'F': [['G', 'G', 'G'], ['G', 'G', 'G'], ['G', 'G', 'G']],
            'B': [['B', 'B', 'B'], ['B', 'B', 'B'], ['B', 'B', 'B']]
        }
        
        cube1 = Cube()
        cube2 = Cube(solved_dict)
        
        # Compare all faces
        for face in 'UDLRFB':
            face1 = cube1._get_face_map(face)
            face2 = cube2._get_face_map(face)
            np.testing.assert_array_equal(face1, face2, 
                                        f"Face {face} should be identical between default and dict initialization")

    def test_sticker_list_vs_solved_equivalence(self):
        """Test that list initialization with solved pattern equals default cube."""
        solved_list = ['W']*9 + ['O']*9 + ['G']*9 + ['R']*9 + ['B']*9 + ['Y']*9
        
        cube1 = Cube()
        cube2 = Cube(solved_list)
        
        # Compare all faces
        for face in 'UDLRFB':
            face1 = cube1._get_face_map(face)
            face2 = cube2._get_face_map(face)
            np.testing.assert_array_equal(face1, face2,
                                        f"Face {face} should be identical between default and list initialization")

    def test_sticker_dict_vs_list_equivalence(self):
        """Test that dictionary and list initialization produce equivalent results."""
        # Create same pattern using both methods
        pattern_dict = {
            'U': [['W', 'R', 'W'], ['G', 'W', 'B'], ['W', 'Y', 'W']],
            'D': [['Y', 'O', 'Y'], ['R', 'Y', 'G'], ['Y', 'W', 'Y']],
            'L': [['O', 'W', 'O'], ['B', 'O', 'Y'], ['O', 'R', 'O']],
            'R': [['R', 'G', 'R'], ['W', 'R', 'O'], ['R', 'B', 'R']],
            'F': [['G', 'Y', 'G'], ['O', 'G', 'R'], ['G', 'W', 'G']],
            'B': [['B', 'R', 'B'], ['Y', 'B', 'W'], ['B', 'G', 'B']]
        }
        
        # Convert to list format: U, L, F, R, B, D
        pattern_list = []
        for face_char in 'ULFRBD':
            face_data = pattern_dict[face_char]
            for row in face_data:
                pattern_list.extend(row)
        
        cube1 = Cube(pattern_dict)
        cube2 = Cube(pattern_list)
        
        # Compare all faces
        for face in 'UDLRFB':
            face1 = cube1._get_face_map(face)
            face2 = cube2._get_face_map(face)
            np.testing.assert_array_equal(face1, face2,
                                        f"Face {face} should be identical between dict and list initialization")


class TestCubeMoveValidation(unittest.TestCase):
    """Tests to validate that all moves work correctly from a solved state."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cube = Cube()
    
    def test_all_basic_moves_from_solved_state(self):
        """Test that all basic face moves work correctly from solved state."""
        basic_moves = ["U", "U'", "U2", "D", "D'", "D2", "F", "F'", "F2", 
                      "B", "B'", "B2", "R", "R'", "R2", "L", "L'", "L2"]
        
        failed_moves = []
        for move in basic_moves:
            with self.subTest(move=move):
                cube = Cube()  # Fresh solved cube for each test
                try:
                    cube.turn(move)
                    # Check that cube is still valid after the move
                    self.assertTrue(cube.is_valid(), f"Move '{move}' created invalid cube state")
                    
                    # Check that no pieces disappeared by checking string output
                    cube_str = str(cube)
                    color_count = sum(1 for char in cube_str if char in 'WYGBRO')
                    expected_colors = 54  # 6 faces * 9 stickers each
                    if color_count != expected_colors:
                        failed_moves.append(f"{move}: expected {expected_colors} colors, got {color_count}")
                        
                except Exception as e:
                    failed_moves.append(f"{move}: {type(e).__name__}: {e}")
        
        if failed_moves:
            self.fail(f"Failed moves from solved state:\n" + "\n".join(failed_moves))
    
    def test_all_wide_moves_from_solved_state(self):
        """Test that all wide moves work correctly from solved state."""
        wide_moves = ["u", "u'", "u2", "d", "d'", "d2", "f", "f'", "f2", 
                     "b", "b'", "b2", "r", "r'", "r2", "l", "l'", "l2"]
        
        failed_moves = []
        for move in wide_moves:
            with self.subTest(move=move):
                cube = Cube()  # Fresh solved cube for each test
                try:
                    cube.turn(move)
                    # Check that cube is still valid after the move
                    self.assertTrue(cube.is_valid(), f"Move '{move}' created invalid cube state")
                    
                    # Check that no pieces disappeared by checking string output
                    cube_str = str(cube)
                    color_count = sum(1 for char in cube_str if char in 'WYGBRO')
                    expected_colors = 54  # 6 faces * 9 stickers each
                    if color_count != expected_colors:
                        failed_moves.append(f"{move}: expected {expected_colors} colors, got {color_count}")
                        
                except Exception as e:
                    failed_moves.append(f"{move}: {type(e).__name__}: {e}")
        
        if failed_moves:
            self.fail(f"Failed wide moves from solved state:\n" + "\n".join(failed_moves))
    
    def test_all_slice_moves_from_solved_state(self):
        """Test that all slice moves work correctly from solved state."""
        slice_moves = ["M", "M'", "M2", "E", "E'", "E2", "S", "S'", "S2"]
        
        failed_moves = []
        for move in slice_moves:
            with self.subTest(move=move):
                cube = Cube()  # Fresh solved cube for each test
                try:
                    cube.turn(move)
                    # Check that cube is still valid after the move
                    self.assertTrue(cube.is_valid(), f"Move '{move}' created invalid cube state")
                    
                    # Check that no pieces disappeared by checking string output
                    cube_str = str(cube)
                    color_count = sum(1 for char in cube_str if char in 'WYGBRO')
                    expected_colors = 54  # 6 faces * 9 stickers each
                    if color_count != expected_colors:
                        failed_moves.append(f"{move}: expected {expected_colors} colors, got {color_count}")
                        
                except Exception as e:
                    failed_moves.append(f"{move}: {type(e).__name__}: {e}")
        
        if failed_moves:
            self.fail(f"Failed slice moves from solved state:\n" + "\n".join(failed_moves))
    
    def test_all_rotation_moves_from_solved_state(self):
        """Test that all cube rotation moves work correctly from solved state."""
        rotation_moves = ["x", "x'", "x2", "y", "y'", "y2", "z", "z'", "z2"]
        
        failed_moves = []
        for move in rotation_moves:
            with self.subTest(move=move):
                cube = Cube()  # Fresh solved cube for each test
                try:
                    cube.turn(move)
                    # Check that cube is still valid after the move
                    self.assertTrue(cube.is_valid(), f"Move '{move}' created invalid cube state")
                    
                    # Check that no pieces disappeared by checking string output
                    cube_str = str(cube)
                    color_count = sum(1 for char in cube_str if char in 'WYGBRO')
                    expected_colors = 54  # 6 faces * 9 stickers each
                    if color_count != expected_colors:
                        failed_moves.append(f"{move}: expected {expected_colors} colors, got {color_count}")
                        
                    # Rotation moves should keep cube solved
                    self.assertTrue(cube.is_solved(), f"Rotation move '{move}' should keep cube solved")
                        
                except Exception as e:
                    failed_moves.append(f"{move}: {type(e).__name__}: {e}")
        
        if failed_moves:
            self.fail(f"Failed rotation moves from solved state:\n" + "\n".join(failed_moves))
    
    def test_comprehensive_move_validation(self):
        """Comprehensive test of all 54 moves to find any that cause issues."""
        all_moves = [
            "U", "U'", "U2", "D", "D'", "D2", "F", "F'", "F2", "B", "B'", "B2", 
            "R", "R'", "R2", "L", "L'", "L2", "u", "u'", "u2", "d", "d'", "d2", 
            "f", "f'", "f2", "b", "b'", "b2", "r", "r'", "r2", "l", "l'", "l2",
            "M", "M'", "M2", "E", "E'", "E2", "S", "S'", "S2", 
            "x", "x'", "x2", "y", "y'", "y2", "z", "z'", "z2"
        ]
        
        # Store detailed information about each failed move
        failed_moves = []
        problem_details = {}
        
        for move in all_moves:
            cube = Cube()  # Fresh solved cube for each test
            try:
                # Get initial state for comparison
                initial_str = str(cube)
                initial_colors = sum(1 for char in initial_str if char in 'WYGBRO')
                
                # Perform the move
                cube.turn(move)
                
                # Get state after move
                after_str = str(cube)
                after_colors = sum(1 for char in after_str if char in 'WYGBRO')
                
                # Check for issues
                issues = []
                
                if not cube.is_valid():
                    issues.append("cube.is_valid() returned False")
                
                if after_colors != initial_colors:
                    issues.append(f"Color count changed: {initial_colors} -> {after_colors}")
                
                # Check for empty spots in the actual color positions (not formatting)
                # Extract just the color characters and check if any are missing
                color_chars = ''.join(char for char in after_str if char in 'WYGBRO')
                if len(color_chars) != 54:
                    issues.append(f"Missing colors in output: expected 54, got {len(color_chars)}")
                
                # Check for any character that's not a valid color or formatting
                valid_chars = set('WYGBRO\n-| ')
                invalid_chars = set(after_str) - valid_chars
                if invalid_chars:
                    issues.append(f"String output contains invalid characters: {invalid_chars}")
                
                if issues:
                    failed_moves.append(move)
                    problem_details[move] = {
                        'issues': issues,
                        'before_str': initial_str,
                        'after_str': after_str,
                        'before_colors': initial_colors,
                        'after_colors': after_colors
                    }
                    
            except Exception as e:
                failed_moves.append(move)
                problem_details[move] = {
                    'issues': [f"Exception: {type(e).__name__}: {e}"],
                    'before_str': initial_str if 'initial_str' in locals() else "N/A",
                    'after_str': "N/A",
                    'before_colors': initial_colors if 'initial_colors' in locals() else 0,
                    'after_colors': 0
                }
        
        # If we found problems, report them in detail
        if failed_moves:
            error_report = ["DETAILED PROBLEM REPORT:"]
            for move in failed_moves:
                details = problem_details[move]
                error_report.append(f"\nMove '{move}':")
                error_report.append(f"  Issues: {', '.join(details['issues'])}")
                error_report.append(f"  Colors before/after: {details['before_colors']}/{details['after_colors']}")
                if len(details['after_str']) < 200:  # Only show if not too long
                    error_report.append(f"  Output after move: {repr(details['after_str'])}")
            
            self.fail(f"Found {len(failed_moves)} problematic moves: {failed_moves}\n" + "\n".join(error_report))

class TestCubeIntegration(unittest.TestCase):
    """Integration tests for cube operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cube = Cube()
    
    def test_sexy_move_pattern(self):
        """Test the famous 'sexy move' pattern R U R' U'."""
        # Apply sexy move 6 times should return to original state
        initial_state = self._get_cube_state()
        
        for _ in range(6):
            self.cube.turn("R U R' U'")
        
        final_state = self._get_cube_state()
        self._assert_states_equal(initial_state, final_state)
    
    def test_wide_move_basic_functionality(self):
        """Test that wide moves work without crashing."""
        # Test that wide moves don't crash and affect the expected slices
        initial_state = self._get_cube_state()
        
        self.cube.turn("r")
        wide_r_state = self._get_cube_state()
        
        # Wide r should affect at least the right face and middle slice
        # We just test that the state changed (not specific position requirements)
        self.assertNotEqual(initial_state, wide_r_state)
    
    def test_scramble_and_solve_attempt(self):
        """Test applying a scramble sequence."""
        scramble = "R U R' U' F' U F U R U2 R' U2 R U' R'"
        self.cube.turn(scramble)
        # Should not crash and cube should be in some scrambled state
        # We can't easily test if it's "correctly scrambled" without 
        # implementing a full solve checker
    
    def _get_cube_state(self):
        """Helper method to get the current state of self.cube."""
        return self._get_cube_state_from_cube(self.cube)
    
    def _get_cube_state_from_cube(self, cube):
        """Helper method to get the state of a given cube."""
        state = {}
        for x, y, z in np.ndindex(3, 3, 3):
            if cube.grid[x, y, z] is not None:
                state[(x, y, z)] = cube.grid[x, y, z].orientation.copy()
        return state
    
    def _assert_states_equal(self, state1, state2):
        """Helper method to assert that two cube states are equal."""
        self.assertEqual(set(state1.keys()), set(state2.keys()))
        for pos in state1:
            self.assertEqual(state1[pos], state2[pos], 
                           f"States differ at position {pos}")


if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2)
