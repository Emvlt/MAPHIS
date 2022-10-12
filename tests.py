import unittest

from utils import constants

class TestConstants(unittest.TestCase):
    def test_feature_in_hex(self):
        for feature_name in constants.FEATURENAMES:
            self.assertIn(feature_name, constants.HEXKEYS)

    def test_feature_in_color_threshold(self):
        for feature_name in constants.FEATURENAMES:
            self.assertIn(feature_name, constants.COLORTHRESHOLD)

class TestPaths(unittest.TestCase):
    def test_is_path_a_folder(self):
        for city_path in constants.CITYPATH.values():
            self.assertTrue(city_path.is_dir())
        for city_path in constants.PROJECTPATHS:
            self.assertTrue(city_path.is_dir())

if __name__ == '__main__':
    unittest.main()