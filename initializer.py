import json
import pandas as pd
import json
import pandas as pd
import string
from transformers import AutoModel, AutoTokenizer


class DataStore:
    def __init__(self):
        #assuming that the files exists
        (self.sentences, self.smixut_list, self.unique_words, self.unique_genders,
         self.unique_pos, self.unique_dep_func, self.unique_words_lex) = self.make_parameters()


    def convert_json_to_dict(self, json_data):
        # Extract the necessary fields from the JSON data
        word = json_data.get("token")
        lex = json_data.get("lex")
        pos = json_data.get("morph", {}).get("pos")
        gender = json_data.get("morph", {}).get("feats", {}).get("Gender")
        dep_func = json_data.get("syntax", {}).get("dep_func")
        # Create the dictionary with the desired keys
        result = {
            "word": word,
            "lex": lex,
            "pos": pos,
            "Gender": gender,
            "dep_func": dep_func
        }
        return result

    def process_json(self, input_json):
        # Extract tokens from the input JSON
        tokens = input_json[0].get("tokens", [])

        # Convert each token to the desired dictionary format
        result = [self.convert_json_to_dict(token) for token in tokens]

        return result

    def add_unique_words_lex_to_dict(self, sentence_list, word_dict):
        punctuation = r"\"#$%&'()*+,-–./:;<=>?@[\]^_`{|}~"
        for item in sentence_list:
            lex_word = item.get("lex")
            if not (len(lex_word) == 1 and lex_word in punctuation) and (lex_word and lex_word not in word_dict):
                word_dict[lex_word] = len(word_dict)
        return word_dict

    def add_unique_words_to_dict(self, sentence_list, word_dict):
        punctuation = r"\"#$%&'()*+,-–./:;<=>?@[\]^_`{|}~"
        for item in sentence_list:
            lex_word = item.get("word")
            if not (lex_word in punctuation) and (lex_word and lex_word not in word_dict):
                word_dict[lex_word] = len(word_dict)
        return word_dict

    def add_unique_pos_to_dict(self, sentence_list, pos_dict):
        punctuation = r"\"#$%&'()*+,-–./:;<=>?@[\]^_`{|}~"
        for item in sentence_list:
            lex_word = item.get("word")
            lex_pos = item.get("pos")
            if not (lex_word in punctuation) and (lex_pos and lex_pos not in pos_dict):
                pos_dict[lex_pos] = len(pos_dict)
        return pos_dict

    def add_unique_dep_func_to_dict(self, sentence_list, dep_func_dict):
        punctuation = r"\"#$%&'()*+,-–./:;<=>?@[\]^_`{|}~"
        for item in sentence_list:
            lex_word = item.get("word")
            lex_dep_func = item.get("dep_func")
            if not (lex_word in punctuation) and (lex_dep_func and lex_dep_func not in dep_func_dict):
                dep_func_dict[lex_dep_func] = len(dep_func_dict)

        print(dep_func_dict)
        return dep_func_dict

    def add_smixut_to_uniqe_words(self, smixut_list, uniqe_words):
        punctuation = r"\"#$%&'()*+,-–./:;<=>?@[\]^_`{|}~"
        for smixut in smixut_list:
            if '-' not in smixut or '–' not in smixut:
                word1, word2 = smixut.split(' ', 1)
                if not (len(word1) == 1 and word1 in punctuation) and (word1 and word1 not in uniqe_words):
                    uniqe_words[word1] = len(uniqe_words)
                if not (len(word2) == 1 and word2 in punctuation) and (word2 and word2 not in uniqe_words):
                    uniqe_words[word2] = len(uniqe_words)

        return uniqe_words

    def make_parameters(self):
        unique_words_lex = {}
        unique_words = {}
        unique_genders = { "Masc", "Fem" }
        unique_pos = self.get_unique_pos()
        unique_dep_func = self.get_unique_dep_func()
        smixut_list = []
        sentences = []
        with open('smixut_file.json', 'r', encoding='utf-8') as smixut_file:
            for smixut in smixut_file:
                smixut = smixut.replace('\n', '').replace('"', '').replace("'", '')
                smixut_list.append(smixut)
            # Read 'output.json' and process the data
        with open('output.json', 'r', encoding='utf-8') as file:
            for line in file:
                if line:
                    data = json.loads(line)
                    lst = self.process_json(data)
                    sentences.append(lst)
                    self.add_unique_words_to_dict(lst, unique_words)
                    self.add_unique_words_lex_to_dict(lst, unique_words_lex)
                    # self.add_unique_pos_to_dict(lst, unique_pos)
                    # self.add_unique_dep_func_to_dict(lst, unique_dep_func)
        self.add_smixut_to_uniqe_words(smixut_list, unique_words)

        return sentences,smixut_list ,unique_words, unique_genders, unique_pos, unique_dep_func, unique_words_lex

    def get_unique_dep_func(self):
        unique_dep_funcs = {
            "nsubj": 0,  # Nominal Subject
            "obj": 1,  # Object
            "obl": 2,  # Oblique
            "advmod": 3,  # Adverbial Modifier
            "acl:relcl": 4,  # Relative Clause Modifier
            "compound:smixut": 5,  # Compound Smixut
            "cop": 6,  # Copula
            "punct": 7,  # Punctuation
            "nmod": 8,
            "appos": 9,
            "conj": 10,
            "case": 11,
            "det": 12,
            "nmod:poss": 13,
            "fixed": 14,
            "amod": 15,
            "parataxis": 16,
            "acl": 17,
            "nummod": 18,
            "ccomp": 19,
            "root": 20,
            "nsubj:cop": 21,
            "obl:tmod": 22,
            "xcomp": 23,
            "orphan": 24
        }

        return unique_dep_funcs

    def get_unique_pos(self):
        # Create a dictionary of unique POS tags with their corresponding index
        unique_pos = {
            "NOUN": 0,  # Noun
            "ADV": 1,  # Adverb
            "NUM": 2,  # Numeral
            "AUX": 3,  # Auxiliary Verb
            "ADJ": 4,  # Adjective
            "PUNCT": 5,  # Punctuation
            "PROPN": 6,  # Proper Noun
            "VERB": 7,  # Verb
            "SCONJ": 8,  # Subordinating Conjunction
            "ADP": 9,
            "DET": 10,
            "CCONJ": 11,
            "X": 12,
            "PRON": 13
        }
        return unique_pos

    def analyze_xl_file(self):
        tokenizer = AutoTokenizer.from_pretrained('dicta-il/dictabert-tiny-joint')
        model = AutoModel.from_pretrained('dicta-il/dictabert-tiny-joint', trust_remote_code=True)
        model.eval()
        file_path = 'Sentence_classification.xlsx'
        df = pd.read_excel(file_path)
        column_name = 'משפט'
        column_sentence = []
        column_smixut = []
        sentences = []
        if column_name in df.columns:
            column_sentence = df[column_name].tolist()
        column_name1 = 'מבנה סמיכות'
        if column_name1 in df.columns:
            column_smixut = df[column_name1].tolist()
            unique_words_lex = {}
            unique_words = {}
            unique_genders = {"Masc", "Fem"}
            unique_pos = {}
            unique_dep_func = {}
            with open('output.json', 'w', encoding='utf-8') as f:
                for sentence in column_sentence:
                     predictions = model.predict([sentence], tokenizer, output_style='json')
                     predictions_json = json.dumps(predictions, ensure_ascii=False)
                     f.write(predictions_json+'\n')
            smixut_list = []
            with open('smixut_file.json', 'w', encoding='utf-8') as smixut_file:
                for smixut in column_smixut:
                    smixut_list.append(smixut)
                    smixut_json = json.dumps(smixut, ensure_ascii=False)
                    smixut_file.write(smixut_json + '\n')
            # Read 'output.json' and process the data

            # add_smixut_to_uniqe_words(smixut_list, unique_words)
            # lst = process_json(data)
            # with open('output.txt', 'w', encoding='utf-8') as file1:
            #     print_to_file(sentences, file1)

            # with open('self.unique_words.txt', 'w', encoding='utf-8') as file:
            #     print_to_file(self.unique_words, file)

            # with open('self.unique_words_lex.txt', 'w', encoding='utf-8') as file:
            #     print_to_file(self.unique_words_lex, file)
