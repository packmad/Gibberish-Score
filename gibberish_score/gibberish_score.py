import math
import pathlib
import pickle
import secrets
import string
import sys
import tempfile

from collections import Counter, defaultdict
from networkx import DiGraph, MultiDiGraph
from os.path import isfile, isdir, join
from statistics import mean, mode, median, stdev, quantiles
from typing import List, Dict


class ProbabilityMarkovChain:

    def __init__(self, top: int = 3):
        self.markov_chain_graph = None
        self.multi_di_graph = MultiDiGraph()
        self.top = top  # top x probability for generating a string

    def training(self, training_txt: str) -> DiGraph:
        assert isfile(training_txt)
        with open(training_txt) as fp:
            words = [line for line in fp.read().splitlines() if len(line) >= 2]
            for w in words:
                self.add_training_string(w)
        return self.build_model()

    def add_training_string(self, s: str):
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
            char = secrets.choice(n_w[:self.top])[0]
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
            return sys.float_info.min
        prob_score = math.exp(log_sum)
        assert 0.0 < prob_score < 1.0
        return prob_score


class GibberishScore:

    def __init__(self, model_pickle: str):
        self.rmc: ProbabilityMarkovChain = pickle.load(open(model_pickle, 'rb'))
        self.threshold = None

    def get_gibberish_score(self, input_string: str) -> float:
        return abs(math.frexp(self.rmc.get_score(input_string))[1])

    def get_nongibberish_string(self, length: int):
        return self.rmc.get_nongibberish_string(length)

    def is_gibberish(self, input_string: str) -> bool:
        if self.threshold is None:
            raise Exception('No threshold given')
        try:
            return self.get_gibberish_score(input_string) > self.threshold[len(input_string)]
        except KeyError:
            return True


def gibberish_score_factory(
        dataset_name: str = 'english',
        model_path: str = None,
        threshold: bool = False) -> GibberishScore:

    parent_folder = pathlib.Path(__file__).parent
    datasets_folder = join(parent_folder, 'datasets')
    assert isdir(datasets_folder)
    words_txt = join(datasets_folder, f'{dataset_name}_words.txt')
    assert isfile(words_txt)
    if model_path is None:
        model_path = join(tempfile.gettempdir(), f'{dataset_name}_words.pickle')
    if not isfile(model_path):
        rmc = ProbabilityMarkovChain()
        rmc.training(words_txt)
        rmc.save_model(model_path)
    gs = GibberishScore(model_path)
    if not threshold:
        return gs
    with open(words_txt) as fp:
        words = {line: gs.get_gibberish_score(line) for line in fp.read().splitlines() if len(line) >= 2}
    len_to_gs = defaultdict(list)
    for k, v in words.items():
        len_to_gs[len(k)].append(v)
    if dataset_name == 'english':
        # precomputed
        gs.threshold = {2: 10, 3: 14, 4: 15, 5: 19, 6: 22, 7: 26, 8: 30, 9: 33, 10: 36, 11: 39, 12: 42, 13: 46, 14: 49,
                        15: 53, 16: 56, 17: 59, 18: 63, 19: 67, 20: 71, 21: 75, 22: 82, 23: 84, 24: 95, 25: 87, 27: 91}
    else:
        gs.threshold = {k: round(quantiles(v, n=10)[8]) for k, v in len_to_gs.items() if len(v) > 2}
    return gs


if __name__ == '__main__':
    pass
