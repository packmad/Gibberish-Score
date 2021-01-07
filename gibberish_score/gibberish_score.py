import math
import pathlib
import pickle
import secrets
import string
import sys
from collections import Counter, defaultdict
from os.path import isfile, join, basename
from random import randint
from statistics import quantiles
from typing import List

from networkx import DiGraph, MultiDiGraph


class ProbabilityMarkovChain:

    def __init__(self, top: int = 5):
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
        if length <= 0:
            raise Exception('Length must be greater than 0')
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
        with open(model_pickle, 'rb') as fp:
            self.pmc: ProbabilityMarkovChain = pickle.load(fp)
        self.threshold = None
        self.deterministic_string_mapping = dict()
        self.deterministic_string_codomain = set()
        self.deterministic_string_codomain.add(None)

    def get_gibberish_score(self, input_string: str) -> float:
        return abs(math.frexp(self.pmc.get_score(input_string))[1])

    def get_nongibberish_string(self, length: int):
        return self.pmc.get_nongibberish_string(length)

    def get_uniq_deterministic_nongibberish_string(self, in_str: str, same_length: bool = False):
        dngs = self.deterministic_string_mapping.get(in_str)
        if dngs is None:
            length = len(in_str)
            if not same_length:
                length = randint(length, length+4)
            while dngs in self.deterministic_string_codomain:
                dngs = list(self.pmc.get_nongibberish_string(length))
                for i in [i for i, c in enumerate(in_str) if c.isupper()]:
                    dngs[i] = dngs[i].upper()
                dngs = ''.join(dngs)
            self.deterministic_string_codomain.add(dngs)
            self.deterministic_string_mapping[in_str] = dngs
        return dngs

    def is_gibberish(self, input_string: str) -> bool:
        if self.threshold is None:
            raise Exception('No threshold given')
        try:
            return self.get_gibberish_score(input_string) > self.threshold[len(input_string)]
        except KeyError:
            return True


def model_builder(dataset_txt: str) -> str:
    assert isfile(dataset_txt)
    pmc = ProbabilityMarkovChain()
    pmc.training(dataset_txt)
    model_pickle = join(pathlib.Path(dataset_txt).parent, f'{basename(dataset_txt)}_model.pickle')
    pmc.save_model(model_pickle)
    assert isfile(model_pickle)
    return model_pickle


def gibberish_score_threshold_factory(dataset_txt: str) -> GibberishScore:
    model_pickle = model_builder(dataset_txt)
    gs = GibberishScore(model_pickle)
    with open(dataset_txt) as fp:
        words = {line: gs.get_gibberish_score(line) for line in fp.read().splitlines() if len(line) >= 2}
    len_to_gs = defaultdict(list)
    for k, v in words.items():
        len_to_gs[len(k)].append(v)
    if 'english_words.txt' in dataset_txt:  # precomputed
        gs.threshold = {2: 10, 3: 14, 4: 15, 5: 19, 6: 22, 7: 26, 8: 30, 9: 33, 10: 36, 11: 39, 12: 42, 13: 46, 14: 49,
                        15: 53, 16: 56, 17: 59, 18: 63, 19: 67, 20: 71, 21: 75, 22: 82, 23: 84, 24: 95, 25: 87, 27: 91}
    else:
        gs.threshold = {k: round(quantiles(v, n=10)[8]) for k, v in len_to_gs.items() if len(v) > 2}
    return gs


if __name__ == '__main__':
    pass