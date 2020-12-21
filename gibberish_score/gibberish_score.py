import math
import pathlib
import pickle
import secrets
import string
import sys

from collections import Counter
from networkx import DiGraph, MultiDiGraph
from os.path import isfile, isdir, join
from typing import List


class ProbabilityMarkovChain:

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

    def get_nongibberish_string(self, length: int) -> str:
        if length < 2:
            raise Exception('Length must be greater than 1')
        model: DiGraph = self.build_model()
        out_str = []
        char = secrets.choice(string.ascii_lowercase)
        out_str.append(char)
        while len(out_str) < length:
            n_w = [(n, model[char][n]['weight']) for n in model[char]]
            n_w.sort(key=lambda tup: tup[1], reverse=True)
            char = secrets.choice(n_w[:3])[0]
            out_str.append(char)
        return ''.join(out_str)

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
        if len_s < 2:
            raise Exception('String length must be greater than 1')
        log_sum = sum([math.log(n) for n in self.get_probability_walk(s) if n > 0.0])
        if log_sum == 0.0:
            raise Exception('Underflow')
        prob_score = math.exp(log_sum)
        assert 0.0 < prob_score < 1.0
        return prob_score


class GibberishScore:

    def __init__(self, model_pickle: str):
        self.rmc: ProbabilityMarkovChain = pickle.load(open(model_pickle, 'rb'))

    def get_gibberish_score(self, input_string: str) -> float:
        return self.rmc.get_score(input_string)

    def get_nongibberish_string(self, length: int):
        return self.rmc.get_nongibberish_string(length)


def gibberish_score_factory(dataset_name: str = 'english') -> GibberishScore:
    datasets_folder = join(pathlib.Path(__file__).parent.parent, 'datasets')
    assert isdir(datasets_folder)
    words_txt = join(datasets_folder, f'{dataset_name}_words.txt')
    assert isfile(words_txt)
    models_folder = join(pathlib.Path(__file__).parent.parent, 'models')
    assert isdir(models_folder)
    words_pickle = join(models_folder, f'{dataset_name}_words.pickle')
    if not isfile(words_pickle):
        rmc = ProbabilityMarkovChain()
        rmc.training(words_txt)
        rmc.save_model(words_pickle)
    return GibberishScore(words_pickle)


def generate_random_string(l: int) -> str:
    return ''.join(secrets.choice(string.ascii_lowercase) for i in range(l))


if __name__ == '__main__':
    #TODO create tests
    gs: GibberishScore = gibberish_score_factory('english')

    test_strings = [
        'lamer',
        'hacker',
        'noob',
        'pro',
        'antani',
    ]
    for ii in range(2,8):
        test_strings.append(gs.get_nongibberish_string(ii))

    test_strings.extend([generate_random_string(len(s)) for s in test_strings])
    for s in test_strings:
        print(f"'{s}'", gs.get_gibberish_score(s))
