# MIT License

# Copyright (c) 2024 The HuggingFace Team

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

# 该模块负责管理 语料库层面 出现的所有指标。有些指标（比如语料库级别的 BLEU 值）并非是在单个样本层面进行计算的，而是基于整个语料库来计算。其中许多此类 聚合指标 取自 EleutherAIHarness。
# 1) matthews_corrcoef: MCC 马修斯相关系数(Matthews Correlation Coefficient), 衡量分类模型预测结果与实际标签之间相关性. 综合考虑了TP/TN/FP/FN的数量，函数输入是一整组的pred和gold，使用sklearn.metrics.matthews_corrcoef实现。
# 2) CorpusLevelF1Score: 语料库级别的F1Score, Sample级别的是处理单个样本，这个是处理一整组的pred和gold。使用sklearn.metrics.f1_score实现。
# 3) CorpusLevelTranslationMetric: 翻译指标，使用sacrebleu包的实现，可选指标有 bleu / chrf / ter。
#                            bleu: Bilingual Evaluation Understudy, 基本原理是n-gram 匹配，即计算pred与gold中 连续的 n 个词或字符 的匹配程度来评估翻译质量。
#                            chrf: Character n-gram F-score, 基于字符 n-gram 的 F 值，综合考虑了召回率和准确率,用于衡量机器翻译结果与参考译文之间的相似程度。
#                            ter:  Translation Error Rate 翻译错误率, TER = 编辑距离 / 参考译文单词数量。
# 4) CorpusLevelPerplexityMetric: 困惑度指标，困惑度表示语言模型对文本的预测能力，衡量语言模型在预测下一个词或字符时的不确定性，通常定义为交叉熵的指数。
#                                 这里计算的是序列上对数概率平均值，而不同选项有不同的归一化和处理方式，如weighted_perplexity，以单词数量作为对数概率平均值的权重；而bits_per_byte则使用比特数进行归一化，并将结果除以以2为底的对数.
"""This module manages all the metrics occurring at the corpus level.
Some metrics (such as corpus BLEU) are not computed at the individual item level, but over all the corpus.
A number of these aggregations come from the EleutherAIHarness
"""
import logging
import math

import numpy as np
import sacrebleu
import sklearn.metrics

from metrics.sample_preparator import (
    GenerativeCorpusMetricInput,
    LogprobCorpusMetricInput,
    PerplexityCorpusMetricInput,
)
from utils import as_list


logger = logging.getLogger(__name__)


# General aggregations
def matthews_corrcoef(items: list[GenerativeCorpusMetricInput]) -> float:
    """Computes the Matthews Correlation Coefficient, using scikit learn ([doc](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html)).

    Args:
        items (list[dict]): List of GenerativeCorpusMetricInput

    Returns:
        float: Score
    """
    golds = [i.golds for i in items]
    preds = [i.preds for i in items]
    return sklearn.metrics.matthews_corrcoef(golds, preds)


class CorpusLevelF1Score:
    def __init__(self, average: str, num_classes: int = 2):
        """Stores the relevant parameters for the task's corpus level f1 score.

        Args:
            average (str): Method to use to compute the f1 score. Can be weighted, macro, micro.
            num_classes (int, optional): Num of possible choice classes. Defaults to 2. If this parameter is above 2, we'll compute multi f1 corpus score
        """
        if average not in ["weighted", "macro", "micro", None]:
            raise ValueError(
                f"A CorpusLevelF1Score must be initialized with weighted, macro, micro, or None as an average function. {average} was used."
            )
        self.average = average
        self.num_classes = num_classes

    def compute(self, items: list[LogprobCorpusMetricInput]):
        """Computes the metric score over all the corpus generated items, by using the scikit learn implementation."""
        golds = [i.golds for i in items]
        preds = [i.preds for i in items]
        # Single f1
        if self.num_classes == 2:
            fscore = sklearn.metrics.f1_score(golds, preds, average=self.average)
            return np.max(fscore)

        # Multi f1
        f1s = []
        for i in range(self.num_classes):
            f1s.append(sklearn.metrics.f1_score(y_true=golds == i, y_pred=preds == i))
        return float(np.mean(f1s))


class CorpusLevelTranslationMetric:
    def __init__(self, metric_type: str):
        """Stores the relevant parameters for a corpus level translation metric.

        Args:
            metric_type (str): Can be any of bleu, chrf, or ter depending on the metric to use.
        """
        if metric_type == "bleu":
            self.metric = sacrebleu.corpus_bleu
        elif metric_type == "chrf":
            self.metric = sacrebleu.corpus_chrf
        elif metric_type == "ter":
            self.metric = sacrebleu.corpus_ter
        else:
            raise ValueError(f"Unknown corpus level translation metric type : {metric_type}")

    def compute(self, items: list[GenerativeCorpusMetricInput]) -> float:
        """Computes the metric score over all the corpus generated items, by using the sacrebleu implementation."""
        golds = [i.golds for i in items]
        preds = []
        for i in items:
            pred = as_list(i.preds)
            if len(pred) > 1:
                logger.info(
                    f"Multiple predictions present, keeping only the first prediction (when computing sacrebleu.{self.metric.__name__})."
                )
            preds.append(pred[0])
        return float(self.metric(hypotheses=preds, references=golds).score)


class CorpusLevelPerplexityMetric:
    def __init__(self, metric_type: str):
        """Stores the relevant parameter for a corpus level perplexity metric.
        Perplexity metrics compute more or less the same thing, which is a variation on the
        average of log-probabilities over a sequence, but the normalization and processing applied
        is different depending on the metric type.
        Perplexity uses an exponential and no weights for the average, weighted perplexity uses an exponential
        and the number of words as weights for the log-prob average, and bits per byte uses the number of bits
        for normalization and divides the results by log(2).

        Args:
            metric_type (str): Can be any of `perplexity`, `weighted_perplexity` or `bits_per_byte`
        """
        if metric_type not in ["perplexity", "weighted_perplexity", "bits_per_byte"]:
            raise ValueError(f"Unknown corpus level perplexity metric type : {metric_type}")

        self.metric_type = metric_type

    def compute(self, items: list[PerplexityCorpusMetricInput]):
        """Computes the metric score over all the corpus generated items."""
        logprobs = [i.logprobs for i in items]
        weights = [i.weights for i in items]

        if self.metric_type == "perplexity":
            return math.exp(-np.mean(logprobs))
        if self.metric_type == "weighted_perplexity":
            return math.exp(-sum(logprobs) / sum(weights))
        if self.metric_type == "bits_per_byte":
            return -sum(logprobs) / sum(weights) * 1 / math.log(2)
