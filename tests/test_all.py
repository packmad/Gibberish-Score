import pathlib
import secrets
import string
import unittest
from os.path import join

from gibberish_score.gibberish_score import GibberishScore, gibberish_score_threshold_factory


class TestStringMethods(unittest.TestCase):

    @staticmethod
    def generate_random_string(length: int) -> str:
        return ''.join(secrets.choice(string.ascii_letters) for i in range(length))

    @classmethod
    def setUpClass(cls):
        parent_folder = pathlib.Path(__file__).parent.parent
        dataset_txt = join(parent_folder, 'datasets', 'english_words.txt')
        cls.gs: GibberishScore = gibberish_score_threshold_factory(dataset_txt)
        cls.random_words = ['qdh', 'hucn', 'pmjsi', 'hdpfuy', 'lmwigdl', 'ckxhfgsy', 'trvtqhqwk']

    def test_score(self):
        gs_score = self.gs.get_gibberish_score
        real_words = ['ash', 'bone', 'bacon', 'arouse', 'blender', 'tolerant', 'congested']
        for i in range(len(real_words)):
            self.assertTrue(gs_score(real_words[i]) < gs_score(self.random_words[i]))

    def test_generate_non_gibberish(self):
        gs_score = self.gs.get_gibberish_score
        nongs_string = self.gs.get_nongibberish_string
        epsilon = 8
        for _ in range(1024):
            for i in range(3, 10):
                rs = self.generate_random_string(i)
                s = nongs_string(i)
                self.assertTrue(len(s) == len(rs))
                gs_s = gs_score(s)
                gs_rs = gs_score(rs)
                self.assertTrue(gs_rs+epsilon > gs_s)

    def test_threshold(self):
        for w in self.random_words:
            self.assertTrue(self.gs.is_gibberish(w))

    def test_stuff(self):
        gs = GibberishScore('/home/simo/PycharmProjects/Gibberish-Score/datasets/english_words.txt_model.pickle')
        for i in range(3, 21):
            print(gs.get_deterministic_nongibberish_string(self.generate_random_string(i)))

if __name__ == '__main__':
    unittest.main()
