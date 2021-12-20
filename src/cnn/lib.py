# -*- coding: utf-8 -*-
"""
@author: MEHRANI Ardalan <ardalan77400@gmail.com>
"""

import os
import numpy as np
import pronouncing
import re
from spellchecker import SpellChecker
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('wordnet')


def get_metrics(cm, list_metrics):
    """Compute metrics from a confusion matrix (cm)
    cm: sklearn confusion matrix
    returns:
    dict: {metric_name: score}

    """
    dic_metrics = {}
    total = np.sum(cm)

    if 'accuracy' in list_metrics:
        out = np.sum(np.diag(cm))
        dic_metrics['accuracy'] = out / total

    if 'pres_0' in list_metrics:
        num = cm[0, 0]
        den = cm[:, 0].sum()
        dic_metrics['pres_0'] = num / den if den > 0 else 0

    if 'pres_1' in list_metrics:
        num = cm[1, 1]
        den = cm[:, 1].sum()
        dic_metrics['pres_1'] = num / den if den > 0 else 0

    if 'recall_0' in list_metrics:
        num = cm[0, 0]
        den = cm[0, :].sum()
        dic_metrics['recall_0'] = num / den if den > 0 else 0

    if 'recall_1' in list_metrics:
        num = cm[1, 1]
        den = cm[1, :].sum()
        dic_metrics['recall_1'] = num / den if den > 0 else 0

    return dic_metrics

CMU_PHONEMES = ['AA', 'AA0', 'AA1', 'AA2', 'AE', 'AE0', 'AE1', 'AE2', 'AH', 'AH0', 'AH1', 'AH2', 'AO', 'AO0', 'AO1',
                'AO2', 'AW', 'AW0', 'AW1', 'AW2', 'AY', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH', 'EH', 'EH0', 'EH1',
                'EH2', 'ER', 'ER0', 'ER1', 'ER2', 'EY', 'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH', 'IH0', 'IH1', 'IH2',
                'IY', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OW0', 'OW1', 'OW2', 'OY', 'OY0',
                'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UH0', 'UH1', 'UH2', 'UW', 'UW0', 'UW1', 'UW2',
                'V', 'W', 'Y', 'Z', 'ZH']

CMU_PHONEMES_NOSTRESS = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 'F', 'G', 'HH',
                         'IH',
                         'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW',
                         'V', 'W', 'Y', 'Z', 'ZH']

SPECIAL_CHARACTER = """0123456789-,;.!?:'"/\|_@#$%^&*~`+=<>()[]{}"""


class CharVectorizer():
    def __init__(self, maxlen=100, no_stress=False, add_space=False, special_character=False, spell_check=False,
                 lemma=False, stem=False):

        self.no_stress = no_stress
        self.add_space = add_space
        self.special_character = special_character
        self.spell_check = spell_check
        if self.spell_check:
            self.spell = SpellChecker()
        self.lemma = lemma
        if self.lemma:
            self.lemmatizer = WordNetLemmatizer()
        self.stem = stem
        if self.stem:
            self.stemmer = PorterStemmer()
        if self.no_stress:
            self.phoneme = CMU_PHONEMES_NOSTRESS
        else:
            self.phoneme = CMU_PHONEMES
        if self.special_character:
            self.phoneme.extend(SPECIAL_CHARACTER)
        self.maxlen = maxlen
        self.phoneme_dict = {k: i for i, k in
                             enumerate(self.phoneme, 1)}  # indice zero is reserved to blank and unknown characters
        print(self.phoneme_dict)

    def transform(self, sentences):
        """
        sentences: list of string
        list of review, review is a list of sequences, sequences is a list of int
        """
        sequences = []
        for sentence in sentences:
            converted = self.convert_to_cmu(sentence)
            seq = [self.phoneme_dict.get(ph, 0) for ph in converted[:self.maxlen]]
            sequences.append(seq)
        return sequences

    def convert_to_cmu(self, sentence):
        transfered = []
        if self.special_character:
            split = re.findall(r"""[\w']+|[0-9-,;.!?:'\"/\\|_@#$%^&*~`+=<>()[\]{}]""", sentence)
        else:
            sentence = re.sub("[^a-zA-Z]+", " ", sentence)
            split = sentence.split()
        for index, w in enumerate(split):
            if self.add_space and index != 0:
                transfered.append(0)
            if self.special_character and not w.isalpha():
                transfered.append(w)
                continue
            if self.lemma:
                w = self.lemmatizer.lemmatize(w)
            if self.stem:
                w = self.stemmer.stem(w)
            phones = pronouncing.phones_for_word(w)
            if self.spell_check and len(phones) == 0:
                phones = pronouncing.phones_for_word(self.spell.correction(w))
            if len(phones) != 0:
                result = phones[0]
                if self.no_stress:
                    result = re.sub("[0-9]+", "", result)
                transfered.extend(result.split())

        return transfered

    def get_params(self):
        params = vars(self)
        return params
