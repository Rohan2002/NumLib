import unittest
import src.matrix as matrix


class MatrixTestCase(unittest.TestCase):
    def test_inverse_matrix(self):
        mat = [[1, 2, 4], [1, 6, 8], [1, 1, 6]]

        true_inv_mat = [
            [7 / 3, -2 / 3, -2 / 3],
            [1 / 6, 1 / 6, -1 / 3],
            [-5 / 12, 1 / 12, 1 / 3],
        ]
        inv_mat = matrix.get_matrix_inverse(mat)

        self.assertAlmostEqual(
            matrix.round_matrix(inv_mat), matrix.round_matrix(true_inv_mat)
        )

        mat_error = [[1, 2, 3]]
        self.assertRaises(ValueError, matrix.get_matrix_inverse, mat_error)


if __name__ == "__main__":
    unittest.main()
