# MIT License

# Copyright (c) 2021 Alex Fabbri, Wojciech Kryściński, Bryan
# McCann, Caiming Xiong, Richard Socher, and Dragomir Radev and The HuggingFace
# Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# pylint: disable=C0103,W0221,W0106
# Replace summ_eval.data_stats_metric
import logging
from collections import Counter
from multiprocessing import Pool

import spacy
from collections import namedtuple as _namedtuple


logger = logging.getLogger(__name__)


_en = None


def normalize(tokens, case=False):
    """

    Lowercases and turns tokens into distinct words.

    """

    return [str(t).lower() if not case else str(t) for t in tokens]


################################################################################


class Fragments:
    Match = _namedtuple("Match", ("summary", "text", "length"))

    def __init__(self, summary, text, case=False):
        # self._tokens = tokenize

        if isinstance(summary, str):
            self.summary = summary.split()
        else:
            self.summary = summary
        if isinstance(text, str):
            self.text = text.split()
        else:
            self.text = text

        self._norm_summary = normalize(self.summary, case)
        self._norm_text = normalize(self.text, case)

        self._match(self._norm_summary, self._norm_text)

    def overlaps(self):
        """

        Return a list of Fragments.Match objects between summary and text.
        This is a list of named tuples of the form (summary, text, length):

            - summary (int): the start index of the match in the summary
            - text (int): the start index of the match in the reference
            - length (int): the length of the extractive fragment

        """

        return self._matches

    def strings(self, min_length=0, summary_base=True):
        """

        Return a list of explicit match strings between the summary and reference.
        Note that this will be in the same format as the strings are input. This is
        important to remember if tokenization is done manually. If tokenization is
        specified automatically on the raw strings, raw strings will automatically
        be returned rather than SpaCy tokenized sequences.

        Arguments:

            - min_length (int): filter out overlaps shorter than this (default = 0)
            - raw (bool): return raw input rather than stringified
                - (default = False if automatic tokenization, True otherwise)
            - summary_base (true): strings are based of summary text (default = True)

        Returns:

            - list of overlaps, where overlaps are strings or token sequences

        """

        # Compute the strings against the summary or the text?

        base = self.summary if summary_base else self.text

        # Generate strings, filtering out strings below the minimum length.

        strings = [base[i : i + length] for i, j, length in self.overlaps() if length > min_length]

        # By default, we just return the tokenization being used.
        # But if they user wants a raw string, then we convert.
        # Mostly, this will be used along with spacy.

        # if self._tokens and raw:

        #    for i, s in enumerate(strings):
        #        strings[i] = str(s)

        # Return the list of strings.

        return strings

    def coverage(self, summary_base=True):
        """
        Return the COVERAGE score of the summary and text.

        Arguments:

            - summary_base (bool): use summary as numerator (default = True)

        Returns:

            - decimal COVERAGE score within [0, 1]
        """

        numerator = sum(o.length for o in self.overlaps())

        if summary_base:
            denominator = len(self.summary)
        else:
            denominator = len(self.text)

        if denominator == 0:
            return 0
        else:
            return numerator / denominator

    def density(self, summary_base=True):
        """

        Return the DENSITY score of summary and text.

        Arguments:

            - summary_base (bool): use summary as numerator (default = True)

        Returns:

            - decimal DENSITY score within [0, ...]

        """

        numerator = sum(o.length**2 for o in self.overlaps())

        if summary_base:
            denominator = len(self.summary)
        else:
            denominator = len(self.text)

        if denominator == 0:
            return 0
        else:
            return numerator / denominator

    def compression(self, text_to_summary=True):
        """

        Return compression ratio between summary and text.

        Arguments:

            - text_to_summary (bool): compute text/summary ratio (default = True)

        Returns:

            - decimal compression score within [0, ...]

        """

        ratio = [len(self.text), len(self.summary)]

        try:
            if text_to_summary:
                return ratio[0] / ratio[1]
            else:
                return ratio[1] / ratio[0]

        except ZeroDivisionError:
            return 0

    def _match(self, a, b):
        """

        Raw procedure for matching summary in text, described in paper.

        """

        self._matches = []

        a_start = b_start = 0

        while a_start < len(a):
            best_match = None
            best_match_length = 0

            while b_start < len(b):
                if a[a_start] == b[b_start]:
                    a_end = a_start
                    b_end = b_start

                    while a_end < len(a) and b_end < len(b) and b[b_end] == a[a_end]:
                        b_end += 1
                        a_end += 1

                    length = a_end - a_start

                    if length > best_match_length:
                        best_match = Fragments.Match(a_start, b_start, length)
                        best_match_length = length

                    b_start = b_end

                else:
                    b_start += 1

            b_start = 0

            if best_match:
                if best_match_length > 0:
                    self._matches.append(best_match)

                a_start += best_match_length

            else:
                a_start += 1

class Metric:
    def evaluate_example(self, summary, reference):
        raise NotImplementedError

    def evaluate_batch(self, summaries, references, aggregate=True):
        raise NotImplementedError


def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])


class DataStatsMetric(Metric):
    def __init__(self, n_gram=3, n_workers=24, case=False, tokenize=True):
        """
        Data Statistics metric
        Makes use of Newsroom code: \
            https://github.com/lil-lab/newsroom/blob/master/newsroom/analyze/fragments.py
        Calculates extractive statistics such as coverage, density, compression as
            defined in Newsroom paper as well as the percentage of novel n-grams in the
            summary vs the input text and the percentage of n-grams in the summary which are
            repeated

        NOTE: these statistics are meant to be calculated with respect to the source text
            (e.g. news article) as opposed to the reference.

        Args:
                :param n_gram: compute statistics for n-grams up to and including this length
                :param n_workers: number of processes to use if using multiprocessing
                :param case: whether to lowercase input before calculating statistics
                :param tokenize: whether to tokenize the input; otherwise assumes that the input
                    is a string of space-separated tokens
        """
        self.n_gram = n_gram
        self.n_workers = n_workers
        self.case = case
        self.tokenize = tokenize

        global _en
        try:
            _en = spacy.load("en_core_web_sm")
        except OSError:
            logger.info("Downloading the spacy en_core_web_sm model\n(don't worry, this will only happen once)")
            from spacy.cli import download

            download("en_core_web_sm")
            _en = spacy.load("en_core_web_sm")

    def evaluate_example(self, summary, input_text):
        if self.tokenize:
            input_text = _en(input_text, disable=["tagger", "parser", "ner", "textcat"])
            input_text = [tok.text for tok in input_text]
            summary = _en(summary, disable=["tagger", "parser", "ner", "textcat"])
            summary = [tok.text for tok in summary]
        fragments = Fragments(summary, input_text, case=self.case)
        coverage = fragments.coverage()
        density = fragments.density()
        compression = fragments.compression()
        score_dict = {"coverage": coverage, "density": density, "compression": compression}
        tokenized_summary = fragments.summary
        tokenized_text = fragments.text
        score_dict["summary_length"] = len(tokenized_summary)
        for i in range(1, self.n_gram + 1):
            input_ngrams = list(find_ngrams(tokenized_text, i))
            summ_ngrams = list(find_ngrams(tokenized_summary, i))
            input_ngrams_set = set(input_ngrams)
            summ_ngrams_set = set(summ_ngrams)
            intersect = summ_ngrams_set.intersection(input_ngrams_set)
            try:
                score_dict[f"percentage_novel_{i}-gram"] = (len(summ_ngrams_set) - len(intersect)) / float(
                    len(summ_ngrams_set)
                )
                ngramCounter = Counter()
                ngramCounter.update(summ_ngrams)
                repeated = [key for key, val in ngramCounter.items() if val > 1]
                score_dict[f"percentage_repeated_{i}-gram_in_summ"] = len(repeated) / float(len(summ_ngrams_set))
            except ZeroDivisionError:
                continue
        return score_dict

    def evaluate_batch(self, summaries, input_texts, aggregate=True):
        corpus_score_dict = Counter()
        p = Pool(processes=self.n_workers)
        results = p.starmap(self.evaluate_example, zip(summaries, input_texts))
        p.close()
        if aggregate:
            [corpus_score_dict.update(x) for x in results]
            for key in corpus_score_dict.keys():
                corpus_score_dict[key] /= float(len(input_texts))
            return corpus_score_dict
        else:
            return results

    @property
    def supports_multi_ref(self):
        return False
