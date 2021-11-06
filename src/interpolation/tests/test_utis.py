import unittest
import src.interpolation.utils as utils

class UtilsTestCase(unittest.TestCase):
    def test_vander_monde(self):
        x_1 = [1,2,3]
        true_vander_x1 = [[1, 1, 1], [1, 2, 4], [1, 3, 9]]
        met_vander_x1 = utils.get_vander_monde_matrix(x_1)
        self.assertEqual(true_vander_x1, met_vander_x1)


        met_vander_x1_N = utils.get_vander_monde_matrix(x_1,N=len(x_1))
        self.assertEqual(true_vander_x1, met_vander_x1_N)

        x_2 = [1,2,3]
        N = 7
        true_vander_x2 = [[1, 1, 1, 1, 1, 1, 1], [1, 2, 4, 8, 16, 32, 64], [1, 3, 9, 27, 81, 243, 729]]
        met_vander_x2 = utils.get_vander_monde_matrix(x_2, N=N)
        self.assertEqual(true_vander_x2, met_vander_x2)

if __name__ == '__main__':
    unittest.main()