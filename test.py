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
	pass

class TestGetEig(unittest.TestCase):
	pass

class TestGetEigPerc(unittest.TestCase):
	pass

class TestProjectImage(unittest.TestCase):
	pass

if __name__ == '__main__':
	unittest.main()
