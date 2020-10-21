import unittest
import numpy as np
from pca import load_and_center_dataset, get_covariance, get_eig, get_eig_perc, project_image, display_image

class TestLoadAndCenterDataset(unittest.TestCase):
	def test_load(self):
		x = load_and_center_dataset('mnist.npy')

		# The dataset needs to have the correct shape
		self.assertEqual(np.shape(x), (2000, 784))

		# The dataset should not be constant-valued
		self.assertNotAlmostEqual(np.max(x) - np.min(x), 0)

	def test_center(self):
		x = load_and_center_dataset('mnist.npy')

		# Each coordinate of our dataset should average to 0
		for i in range(np.shape(x)[1]):
			self.assertAlmostEqual(np.sum(x[:, i]), 0)

class TestGetCovariance(unittest.TestCase):
	def test_shape(self):
		x = load_and_center_dataset('mnist.npy')
		S = get_covariance(x)

		# S should be square and have side length d
		self.assertEqual(np.shape(S), (784, 784))

	def test_values(self):
		x = load_and_center_dataset('mnist.npy')
		S = get_covariance(x)

		# S should be symmetric
		self.assertTrue(np.all(np.isclose(S, S.T)))

		# S should have non-negative values on the diagonal
		self.assertTrue(np.min(np.diagonal(S)) >= 0)

class TestGetEig(unittest.TestCase):
	def test_small(self):
		x = load_and_center_dataset('mnist.npy')
		S = get_covariance(x)
		Lambda, U = get_eig(S, 2)

		self.assertEqual(np.shape(Lambda), (2, 2))
		self.assertTrue(np.all(np.isclose(Lambda, [[350880.76329673, 0], [0, 245632.27295307]])))

		# The eigenvectors should be the columns
		self.assertEqual(np.shape(U), (784, 2))
		self.assertTrue(np.all(np.isclose(S @ U, U @ Lambda)))

	def test_large(self):
		x = load_and_center_dataset('mnist.npy')
		S = get_covariance(x)
		Lambda, U = get_eig(S, 784)

		self.assertEqual(np.shape(Lambda), (784, 784))
		# Check that Lambda is diagonal
		self.assertEqual(np.count_nonzero(Lambda - np.diag(np.diagonal(Lambda))), 0)
		# Check that Lambda is sorted in decreasing order
		self.assertTrue(np.all(np.equal(np.diagonal(Lambda), sorted(np.diagonal(Lambda), reverse=True))))

		# The eigenvectors should be the columns
		self.assertEqual(np.shape(U), (784, 784))
		self.assertTrue(np.all(np.isclose(S @ U, U @ Lambda)))

class TestGetEigPerc(unittest.TestCase):
	def test_small(self):
		x = load_and_center_dataset('mnist.npy')
		S = get_covariance(x)
		Lambda, U = get_eig_perc(S, .07)

		self.assertEqual(np.shape(Lambda), (2, 2))
		self.assertTrue(np.all(np.isclose(Lambda, [[350880.76329673, 0], [0, 245632.27295307]])))

		# The eigenvectors should be the columns
		self.assertEqual(np.shape(U), (784, 2))
		self.assertTrue(np.all(np.isclose(S @ U, U @ Lambda)))

	def test_large(self):
		x = load_and_center_dataset('mnist.npy')
		S = get_covariance(x)
		# This will select all eigenvalues/eigenvectors
		Lambda, U = get_eig_perc(S, -1)

		self.assertEqual(np.shape(Lambda), (784, 784))
		# Check that Lambda is diagonal
		self.assertEqual(np.count_nonzero(Lambda - np.diag(np.diagonal(Lambda))), 0)
		# Check that Lambda is sorted in decreasing order
		self.assertTrue(np.all(np.equal(np.diagonal(Lambda), sorted(np.diagonal(Lambda), reverse=True))))

		# The eigenvectors should be the columns
		self.assertEqual(np.shape(U), (784, 784))
		self.assertTrue(np.all(np.isclose(S @ U, U @ Lambda)))

class TestProjectImage(unittest.TestCase):
	def test_shape(self):
		x = load_and_center_dataset('mnist.npy')
		S = get_covariance(x)
		_, U = get_eig(S, 2)
		projected = project_image(x[3], U) # This is the image of the "9" in the spec

		self.assertEqual(np.shape(projected), (784, 1))
		self.assertAlmostEqual(np.min(projected), -113.79455198736488)
		self.assertAlmostEqual(np.max(projected), 120.0658469887994)

if __name__ == '__main__':
	unittest.main()
