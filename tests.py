import unittest

from utils import constants

class TestConstants(unittest.TestCase):
    def test_feature_in_hex(self):
        for feature_name in constants.FEATURENAMES:
            self.assertIn(feature_name, constants.HEXKEYS)

    def test_feature_in_color_threshold(self):
        for feature_name in constants.FEATURENAMES:
            self.assertIn(feature_name, constants.COLORTHRESHOLD)


if __name__ == '__main__':
    unittest.main()
