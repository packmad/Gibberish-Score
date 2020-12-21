from os.path import abspath, dirname, isfile, isdir, join

from networkx import DiGraph, MultiDiGraph
from typing import List, Tuple, Optional
from collections import Counter
import pathlib
import pickle
import string
import secrets
import math
import sys


class RandomnessMarkovChain:

    def __init__(self):
        self.markov_chain_graph = None
        self.multi_di_graph = MultiDiGraph()

    def training(self, training_txt: str) -> DiGraph:
        assert isfile(training_txt)
        with open(training_txt) as fp:
            words = [line for line in fp.read().splitlines() if len(line) > 1]
            for w in words:
                self.add_training_string(w)
        return self.build_model()

    def add_training_string(self, s: str):
        assert len(s) > 1
        s = s.lower()
        for i in range(len(s) - 1):
            self.multi_di_graph.add_edge(s[i], s[i + 1])

    def force_build_model(self) -> DiGraph:
        self.markov_chain_graph = None
        return self.build_model()

    def build_model(self) -> DiGraph:
        if self.markov_chain_graph is not None:
            return self.markov_chain_graph
        self.markov_chain_graph = DiGraph()
        for n in self.multi_di_graph.nodes():
            out_edges_n = self.multi_di_graph.out_edges(n)
            len_out_edges_n = len(out_edges_n)
            out_nodes_list = [oe[1] for oe in out_edges_n]
            if len_out_edges_n > 0:
                perc_counter = dict()
                for k, v in Counter(out_nodes_list).items():
                    perc_counter[k] = v / len_out_edges_n
                assert 0.9 <= sum(perc_counter.values()) <= 1.1
                for k, v in perc_counter.items():
                    self.markov_chain_graph.add_edge(n, k, weight=v)
        return self.markov_chain_graph

    def save_model(self, file_name: str):
        if self.build_model() is not None:
            pickle.dump(self, open(file_name, 'wb'))
        else:
            raise Exception('Logic flawed')

    def get_probability_walk(self, s: str) -> List[float]:
        s = s.lower()
        model: DiGraph = self.build_model()
        probwalk = list()
        for i in range(len(s) - 1):
            try:
                w = model[s[i]][s[i + 1]]['weight']
                assert 0.0 <= w <= 1.0
                probwalk.append(w)
            except KeyError:
                probwalk.append(0.0)
        return probwalk

    def get_score(self, s: str) -> float:
        len_s = len(s)
        if 0 >= len_s >= 1:
            raise Exception('String length must be greater than 1')
        log_sum = sum([math.log(n) for n in self.get_probability_walk(s) if n > 0.0])
        if log_sum == 0.0:
            raise Exception('Underflow')
        prob_score = math.exp(log_sum)
        assert 0.0 < prob_score < 1.0
        return prob_score


class GibberishScore:

    def __init__(self, model_pickle: str):
        self.rmc: RandomnessMarkovChain = pickle.load(open(model_pickle, 'rb'))

    def get_gibberish_score(self, input_string: str) -> float:
        return self.rmc.get_score(input_string)


def gibberish_score_factory() -> GibberishScore:
    datasets_folder = join(pathlib.Path(__file__).parent.parent, 'datasets')
    assert isdir(datasets_folder)
    english_words_txt = join(datasets_folder, 'english_words.txt')
    assert isfile(english_words_txt)
    models_folder = join(pathlib.Path(__file__).parent.parent, 'models')
    assert isdir(models_folder)
    eng_words_pickle = join(models_folder, 'english_words.pickle')
    if not isfile(eng_words_pickle):
        rmc = RandomnessMarkovChain()
        rmc.training(english_words_txt)
        rmc.save_model(eng_words_pickle)
    return GibberishScore(eng_words_pickle)


def generate_random_string(l: int) -> str:
    secrets_gen = secrets.SystemRandom()
    alphabet = string.ascii_letters
    rnd_str = ''.join(secrets.choice(alphabet) for i in range(l))
    return rnd_str


if __name__ == '__main__':
    #TODO create tests
    gs: GibberishScore = gibberish_score_factory()

    test_strings = [
        'lamer',
        'hacker',
        'noob',
        'pro',
        'antani',
    ]

    test_strings.extend([generate_random_string(len(s)) for s in test_strings])
    for s in test_strings:
        print(f"'{s}'", gs.get_gibberish_score(s))
