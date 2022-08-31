import unittest
from ipp_toolkit.data.structured_2d import Uniform2D


class TestData(unittest.TestCase):
    def test_uniform(self):
        uniform = Uniform2D(value=1)
        sample = uniform.sample((0, 0))
        self.assertEqual(sample, 1)


if __name__ == "__main__":
    unittest.main()
