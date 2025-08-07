import unittest
import sys
import os
import numpy as np

# Add the parent directory to the Python path so we can import cuber and cubie
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        face_dict = self.cube.get_face('U')
        self.assertIn('U', face_dict)
        self.assertEqual(len(face_dict['U']), 3)  # 3 rows
        self.assertEqual(len(face_dict['U'][0]), 3)  # 3 columns
        
        # All squares should be 'W' on solved cube's up face
        for row in face_dict['U']:
            for square in row:
                self.assertEqual(square, 'W')
        
        # Test multiple faces
        multiple_faces = self.cube.get_face('UDL')
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
        all_faces = self.cube.get_face('UDLRFB')
        self.assertEqual(len(all_faces), 6)
        
        expected_colors = {'U': 'W', 'D': 'Y', 'L': 'O', 'R': 'R', 'F': 'G', 'B': 'B'}
        
        for face, expected_color in expected_colors.items():
            self.assertIn(face, all_faces)
            for row in all_faces[face]:
                for square in row:
                    self.assertEqual(square, expected_color)
    
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
