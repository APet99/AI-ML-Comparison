import unittest

from utils.preprocess_models_csv import load_model_timings


class TestCSVImport(unittest.TestCase):

    def test_file_exists(self):
        return_state = load_model_timings()
        if return_state == None:
            self.fail()
        else:
            self.assertFalse(return_state, None)


if __name__ == '__main__':
    unittest.main()
