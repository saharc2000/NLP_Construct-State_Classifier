
import json
import pandas as pd
import string
from transformers import AutoModel, AutoTokenizer
from initializer import DataStore

class vector:
    def __init__(self):

        data = DataStore()
        self.sentences = data.sentences
        self.smixut_list = data.smixut_list
        self.unique_words = data.unique_words
        self.unique_genders = data.unique_genders
        self.unique_pos = data.unique_pos
        self.unique_dep_func = data.unique_dep_func
        self.unique_words_lex = data.unique_words_lex

    def find_token_info(self,data, search_token):
        for entry in data:
            for token_info in entry['tokens']:
                if token_info['token'] == search_token:
                    return token_info
        return None


    def print_to_file(self,token_info, file):
        if token_info:
            # with open('output.txt', 'w', encoding='utf-8') as file:
            file.write(json.dumps(token_info, ensure_ascii=False, indent=4))
        else:
            print(f"Token not found.")


    def get_lex(self,word,sentence):
        for item in sentence:
            if item.get("word") == word:
                return item.get("lex")
        return "error"

    def find_string_in_dict_list(self,sentence, target):
        for word_dict in sentence:
            if target in word_dict.get("word"):
                return word_dict.get("lex")
        return "err"  # Return -1 if the target string is not found

    #  Finds and returns a dictionary from the list `sentence` based on a search for a substring in the `word` field
    #     and a positional shift.
    #
    #     Args:
    #         sentence (list of dict): A list of dictionaries where each dictionary contains at least a "word" key.
    #         shift (int): The number of positions to shift from the found item in the `sentence` list.
    #         smixut (str): The substring to search for within the "word" field of each dictionary.
    #
    #     Returns:
    #         dict: The dictionary from the list `sentence` at the position `i + shift`, where `i` is the index of the
    #               dictionary containing the `smixut` substring in its "word" field. If no matching dictionary is found
    #               or the shift goes out of bounds, returns "err".
    def find_dict_in_shift_from_smixut(self,sentence, shift, smixut):
        i=0
        punctuation = r"\"#$%&'()*+,-–./:;<=>?@[\]^_`{|}~"
        for word_dict in sentence:
            if (smixut in word_dict.get("word")) and (0 < i+shift < len(sentence)):
                if(sentence[i+shift].get("word") in punctuation):
                    if(shift > 0):
                        return self.find_dict_in_shift_from_smixut(sentence, shift + 1, smixut)
                    else:
                        return self.find_dict_in_shift_from_smixut(sentence, shift - 1, smixut)
                else:
                    return sentence[i+shift]
            i+=1
        return "err"


    def find_pos_in_dict_list(self,sentence, target):
        for word_dict in sentence:
            if target in word_dict.get("word"):
                return word_dict.get("pos")
        return "err"  # Return -1 if the target string is not found

    def find_dep_in_dict_list(self,sentence, target):
        for word_dict in sentence:
            if target in word_dict.get("word"):
                return word_dict.get("dep_func")
        return "err"  # Return -1 if the target string is not found


       # Returns the key associated with the given value in a dictionary.

       # Args:
         #   d (dict): The dictionary to search.
          #  target_value: The value to find the key for.

       # Returns:
         #   The key associated with the target_value if found, else None.

    def get_key_by_value(self,dict, target_value):
        for key, value in dict.items():
            if key == target_value:
                return value
        return None


    def find_smixut_location_in_sentence(self,smixut, sentence):
        word1, word2 = smixut.split(' ', 1)
        ind = 0
        for word_dict in sentence:
            if word1 == word_dict.get("word"):
                break
            ind += 1
        return ind

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
            "SCONJ": 8  # Subordinating Conjunction
        }
        return unique_pos

    def get_unique_dep_func(self):
        unique_dep_func = {
            "nsubj": 0,  # Nominal Subject
            "obj": 1,  # Object
            "obl": 2,  # Oblique
            "advmod": 3,  # Adverbial Modifier
            "acl:relcl": 4,  # Relative Clause Modifier
            "compound:smixut": 5,  # Compound Smixut
            "cop": 6,  # Copula
            "punct": 7  # Punctuation
        }

        return unique_dep_func

    #vector that uses the word of the smixut
    def make_vector1(self):
        print("[Vector1]:")
        #create list of vectors
        vectors_type1 = []
        self.unique_words = self.unique_words
        for smixut in self.smixut_list:
            word1, word2 = smixut.split(' ', 1)
            # find_token_info()
            vector = [0] * len(self.unique_words)
            # complete get_lex
            vector[self.unique_words[word1]] = 1
            vector[self.unique_words[word2]] = 1
            vectors_type1.append(vector)
        return vectors_type1

    #vector that uses the lex of the smixut
    def make_vector2(self):
        print("[Vector2]:")
        #create list of vectors
        vectors_type2 = []
        i = 0
        self.sentences = self.sentences
        for smixut in self.smixut_list:
            if(i >= len(self.sentences)):
                break
            word1, word2 = smixut.split(' ', 1)
            lex_word1 = self.find_string_in_dict_list(self.sentences[i], word1)
            lex_word2 = self.find_string_in_dict_list(self.sentences[i], word2)
            # find_token_info()
            vector = [0] * len(self.unique_words_lex)
            # complete get_lex
            if(lex_word1 != "err"):
                vector[self.unique_words_lex[lex_word1]] = 1
            if (lex_word2 != "err"):
                vector[self.unique_words_lex[lex_word2]] = 1
            vectors_type2.append(vector)
            i += 1

        return vectors_type2

    #vector that marks all the words in the sentence besides the smixut
    #assuming that the smixut is the word where "dep_func": "compound:smixut" and the previous one
    def make_vector3(self):
        print("[Vector3]:")
        vectors_type3 = []
        punctuation = r"\"#$%&'()*+,-–./:;<=>?@[\]^_`{|}~"
        for i, sentence in enumerate(self.sentences):
            if i >= len(self.smixut_list):
                break
            curr_vec = [0] * len(self.unique_words)
            word1, word2 = self.smixut_list[i].split(' ', 1)
            for word_dict in sentence:
                curr_word = word_dict.get("word")
                if curr_word not in punctuation:
                    if not (curr_word == word1 or curr_word == word2):
                        curr_vec[self.unique_words[curr_word]] = 1
            vectors_type3.append(curr_vec)
        return vectors_type3

    #vector that uses the pos of the smixut
    def make_vector4(self):
        print("[Vector4]:")
        vector_type4 = []
        i=0
        for smixut in self.smixut_list:
            if(i >= len(self.sentences)):
                break
            word1, word2 = smixut.split(' ', 1)
            pos_word1 = self.find_pos_in_dict_list(self.sentences[i], word1)
            pos_word2 = self.find_pos_in_dict_list(self.sentences[i], word2)
            # find_token_info()
            vector = [0] * len(self.unique_pos)
            # complete get_lex
            if(pos_word1 != "err"):
                vector[self.unique_pos[pos_word1]] = 1
            if (pos_word2 != "err"):
                vector[self.unique_pos[pos_word2]] = 1
            vector_type4.append(vector)
            i += 1
        return vector_type4

    #vector that uses the dep_func of the smixut
    def make_vector5(self):
        print("[Vector5]:")
        vector_type5 = []
        i=0
        for smixut in self.smixut_list:
            if(i >= len(self.sentences)):
                break
            word1, word2 = smixut.split(' ', 1)
            dep_word1 = self.find_dep_in_dict_list(self.sentences[i], word1)
            dep_word2 = self.find_dep_in_dict_list(self.sentences[i], word2)
            # find_token_info()
            vector = [0] * len(self.unique_dep_func)
            # complete get_lex
            if(dep_word1 != "err"):
                vector[self.unique_dep_func[dep_word1]] = 1
            if (dep_word2 != "err"):
                vector[self.unique_dep_func[dep_word2]] = 1
            vector_type5.append(vector)
            i += 1
        return vector_type5

    #vector that uses the word of the one word before and after the smixut
    def make_vector6(self):
        print("[Vector6]:")
        #create list of vectors
        vectors_type6 = []
        i = 0
        for smixut in self.smixut_list:
            word1, word2 = smixut.split(' ', 1)
            # find_token_info()
            if (i >= len(self.sentences)):
                break
            vector = [0] * len(self.unique_words)
            word_before = self.find_dict_in_shift_from_smixut(self.sentences[i], -1, word1)
            word_after = self.find_dict_in_shift_from_smixut(self.sentences[i], 1, word2)
            if((word_before != "err")  ):
                vector[self.unique_words[word_before.get("word")]] = 1
            if ((word_after != "err")  ):
                vector[self.unique_words[word_after.get("word")]] = 1
            vectors_type6.append(vector)
            i+=1
        return vectors_type6

    #vector that uses the word before and after smixut with pos and dep func
    def make_vector7(self):
        print("[Vector7]:")
        # Create a list to store the resulting vectors
        pos_dep_func_vectors = []

        for i, smixut in enumerate(self.smixut_list):
            if i >= len(self.sentences):
                break

            word1, word2 = smixut.split(' ', 1)

            # Initialize the vector with zeros
            vector = [0] * (2 * len(self.unique_pos) + 2 * len(self.unique_dep_func))

            # Process the word before the smixut
            word_before = self.find_dict_in_shift_from_smixut(self.sentences[i], -1, word1)
            if word_before != "err":
                pos_before = word_before.get("pos")
                dep_func_before = word_before.get("dep_func")
                if pos_before in self.unique_pos:
                    vector[self.unique_pos[pos_before]] = 1
                if dep_func_before in self.unique_dep_func:
                    vector[len(self.unique_pos) + self.unique_dep_func[dep_func_before]] = 1

            # Process the word after the smixut
            word_after = self.find_dict_in_shift_from_smixut(self.sentences[i], 1, word2)
            if word_after != "err":
                pos_after = word_after.get("pos")
                dep_func_after = word_after.get("dep_func")
                if pos_after in self.unique_pos:
                    vector[len(self.unique_pos) * 2 + self.unique_pos[pos_after]] = 1
                if dep_func_after in self.unique_dep_func:
                    vector[len(self.unique_pos) * 2 + len(self.unique_dep_func) + self.unique_dep_func[dep_func_after]] = 1

            # Append the vector to the list of vectors
            pos_dep_func_vectors.append(vector)

        return pos_dep_func_vectors

    #vector that uses the word of 2 words before and after the smixut
    def make_vector8(self):
        punctuation = r"\"#$%&'()*+,-–./:;<=>?@[\]^_`{|}~"
        #create list of vectors
        vectors_type8 = []
        i = 0
        for smixut in self.smixut_list:
            word1, word2 = smixut.split(' ', 1)
            # find_token_info()
            if (i >= len(self.sentences)):
                break
            vector = [0] * len(self.unique_words)
            one_word_before = self.find_dict_in_shift_from_smixut(self.sentences[i], -1, word1)
            two_words_before = self.find_dict_in_shift_from_smixut(self.sentences[i], -2, word1)
            one_word_after = self.find_dict_in_shift_from_smixut(self.sentences[i], 1, word2)
            two_words_after = self.find_dict_in_shift_from_smixut(self.sentences[i], 2, word2)
            if ((one_word_before != "err") and (not (one_word_before.get("word") in punctuation))):
                vector[self.unique_words[one_word_before.get("word")]] = 1
            if ((two_words_before != "err") and (not (two_words_before.get("word") in punctuation)) ):
                vector[self.unique_words[two_words_before.get("word")]] = 1
            if ((one_word_after != "err") and (not (one_word_after.get("word") in punctuation))):
                vector[self.unique_words[one_word_after.get("word")]] = 1
            if ((two_words_after != "err") and (not (two_words_after.get("word") in punctuation))):
                vector[self.unique_words[two_words_after.get("word")]] = 1

            vectors_type8.append(vector)
            i += 1
        return vectors_type8

    # vector that for each word in the smixut puts the number that represents its dep_func
    def make_vector9(self):
        #create list of vectors
        vectors_type9 = []
        i=0
        for smixut in self.smixut_list:
            if (i >= len(self.sentences)):
                break
            word1, word2 = smixut.split(' ', 1)
            # find_token_info()
            vector = [0] * len(self.unique_words)
            dict1 = self.find_dict_in_shift_from_smixut(self.sentences[i], 0, word1)
            dict2 = self.find_dict_in_shift_from_smixut(self.sentences[i], 0, word2)
            # target_value is the index of word1's dep_func in self.unique_dep_func
            if dict1 != "err":
                vector[self.unique_words[word1]] = self.get_key_by_value(self.unique_dep_func, dict1.get("dep_func"))
            if dict2 != "err":
                vector[self.unique_words[word2]] = self.get_key_by_value(self.unique_dep_func, dict2.get("dep_func"))
            vectors_type9.append(vector)
            i+=1
        return vectors_type9

    #vector that uses the lex of the one word before and after the smixut
    def make_vector10(self):
        #create list of vectors
        vectors_type10 = []
        i = 0
        for smixut in self.smixut_list:
            if (i >= len(self.sentences)):
                break
            word1, word2 = smixut.split(' ', 1)
            # find_token_info()
            if (i >= len(self.sentences)):
                break
            vector = [0] * len(self.unique_words)
            word_before = self.find_dict_in_shift_from_smixut(self.sentences[i], -1, word1)
            word_after = self.find_dict_in_shift_from_smixut(self.sentences[i], 1, word2)
            if((word_before != "err")  ):
                vector[self.unique_words[word_before.get("lex")]] = 1
            if ((word_after != "err")  ):
                vector[self.unique_words[word_after.get("lex")]] = 1
            vectors_type10.append(vector)
            i+=1
        return vectors_type10
    # vector that for each word in the smixut puts the number that represents its lex
    def make_vector11(self):
        #create list of vectors
        vectors_type11 = []
        i=0
        for smixut in self.smixut_list:
            if (i >= len(self.sentences)):
                break
            word1, word2 = smixut.split(' ', 1)
            # find_token_info()
            vector = [0] * len(self.unique_words)
            dict1 = self.find_dict_in_shift_from_smixut(self.sentences[i], 0, word1)
            dict2 = self.find_dict_in_shift_from_smixut(self.sentences[i], 0, word2)
            # target_value is the index of word1's dep_func in self.unique_dep_func
            if dict1 != "err":
                vector[self.unique_words[word1]] = self.get_key_by_value(self.unique_words_lex, dict1.get("lex"))
            if dict2 != "err":
                vector[self.unique_words[word2]] = self.get_key_by_value(self.unique_words_lex, dict2.get("lex"))
            vectors_type11.append(vector)
            i+=1
        return vectors_type11

    #vector that for one word before and one after the smixut puts the number that represents its lex
    def make_vector12(self):
        #create list of vectors
        vectors_type12 = []
        i=0
        for smixut in self.smixut_list:
            if (i >= len(self.sentences)):
                break
            word1, word2 = smixut.split(' ', 1)
            # find_token_info()
            vector = [0] * len(self.unique_words)
            dict1 = self.find_dict_in_shift_from_smixut(self.sentences[i], -1, word1)
            dict2 = self.find_dict_in_shift_from_smixut(self.sentences[i], 1, word2)
            # target_value is the index of word1's dep_func in self.unique_dep_func
            if dict1 != "err":
                vector[self.unique_words[dict1.get("word")]] = self.get_key_by_value(self.unique_words_lex, dict1.get("lex"))
            if dict2 != "err":
                vector[self.unique_words[dict2.get("word")]] = self.get_key_by_value(self.unique_words_lex, dict2.get("lex"))
            vectors_type12.append(vector)
            i+=1
        return vectors_type12


    #vector that for one word before and one after the smixut takes its pos puts the number that represents its lex
    def make_vector13(self):
        #create list of vectors
        vectors_type13 = []
        i=0
        for smixut in self.smixut_list:
            if (i >= len(self.sentences)):
                break
            word1, word2 = smixut.split(' ', 1)
            # find_token_info()
            vector = [0] * len(self.unique_pos)
            dict1 = self.find_dict_in_shift_from_smixut(self.sentences[i], -1, word1)
            dict2 = self.find_dict_in_shift_from_smixut(self.sentences[i], 1, word2)
            if dict1 != "err":
                vector[self.unique_pos[dict1.get("pos")]] = self.get_key_by_value(self.unique_words_lex, dict1.get("lex"))
            if dict2 != "err":
                vector[self.unique_pos[dict2.get("pos")]] = self.get_key_by_value(self.unique_words_lex, dict2.get("lex"))
            vectors_type13.append(vector)
            i+=1
        return vectors_type13

    #vector that uses the word of the two words after the smixut
    def make_vector14(self):
        #create list of vectors
        vectors_type14 = []
        i = 0
        for smixut in self.smixut_list:
            word1, word2 = smixut.split(' ', 1)
            # find_token_info()
            if (i >= len(self.sentences)):
                break
            vector = [0] * len(self.unique_words)
            word_after = self.find_dict_in_shift_from_smixut(self.sentences[i], 1, word2)
            word_after2 = self.find_dict_in_shift_from_smixut(self.sentences[i], 2, word2)
            if((word_after != "err")  ):
                vector[self.unique_words[word_after.get("word")]] = 1
            if ((word_after2 != "err")  ):
                vector[self.unique_words[word_after2.get("word")]] = 1
            vectors_type14.append(vector)
            i+=1
        return vectors_type14


    #vector that uses the lex of the two words after the smixut
    def make_vector15(self):
        #create list of vectors
        vectors_type15 = []
        i = 0
        for smixut in self.smixut_list:
            word1, word2 = smixut.split(' ', 1)
            # find_token_info()
            if (i >= len(self.sentences)):
                break
            vector = [0] * len(self.unique_words_lex)
            word_after = self.find_dict_in_shift_from_smixut(self.sentences[i], 1, word2)
            word_after2 = self.find_dict_in_shift_from_smixut(self.sentences[i], 2, word2)
            if ((word_after != "err")):
                vector[self.unique_words_lex[word_after.get("lex")]] = 1
            if ((word_after2 != "err")):
                vector[self.unique_words_lex[word_after2.get("lex")]] = 1
            vectors_type15.append(vector)
            i+=1
        return vectors_type15

    #vector that for two words after the smixut puts the number that represents its lex in the word part vector

    def make_vector16(self):
        #create list of vectors
        vectors_type16 = []
        i=0
        for smixut in self.smixut_list:
            if (i >= len(self.sentences)):
                break
            word1, word2 = smixut.split(' ', 1)
            # find_token_info()
            vector = [0] * len(self.unique_words)
            dict1 = self.find_dict_in_shift_from_smixut(self.sentences[i], 1, word2)
            dict2 = self.find_dict_in_shift_from_smixut(self.sentences[i], 2, word2)
            # target_value is the index of word1's dep_func in self.unique_dep_func
            if dict1 != "err":
                vector[self.unique_words[dict1.get("word")]] = self.get_key_by_value(self.unique_words_lex, dict1.get("lex"))
            if dict2 != "err":
                vector[self.unique_words[dict2.get("word")]] = self.get_key_by_value(self.unique_words_lex, dict2.get("lex"))
            vectors_type16.append(vector)
            i+=1
        return vectors_type16

    #vector that uses the word of the two words before the smixut
    def make_vector17(self):
        #create list of vectors
        vectors_type17 = []
        i = 0
        for smixut in self.smixut_list:
            word1, word2 = smixut.split(' ', 1)
            # find_token_info()
            if (i >= len(self.sentences)):
                break
            vector = [0] * len(self.unique_words)
            word_before = self.find_dict_in_shift_from_smixut(self.sentences[i], -1, word1)
            word_before2 = self.find_dict_in_shift_from_smixut(self.sentences[i], -2, word1)
            if((word_before != "err")  ):
                vector[self.unique_words[word_before.get("word")]] = 1
            if ((word_before2 != "err")  ):
                vector[self.unique_words[word_before2.get("word")]] = 1
            vectors_type17.append(vector)
            i+=1
        return vectors_type17


    #vector that uses the lex of the two words before the smixut
    def make_vector18(self):
        #create list of vectors
        vectors_type18 = []
        i = 0
        for smixut in self.smixut_list:
            word1, word2 = smixut.split(' ', 1)
            # find_token_info()
            if (i >= len(self.sentences)):
                break
            vector = [0] * len(self.unique_words_lex)
            word_before = self.find_dict_in_shift_from_smixut(self.sentences[i], -1, word1)
            word_before2 = self.find_dict_in_shift_from_smixut(self.sentences[i], -2, word1)
            if ((word_before != "err")):
                vector[self.unique_words_lex[word_before.get("lex")]] = 1
            if ((word_before2 != "err")):
                vector[self.unique_words_lex[word_before2.get("lex")]] = 1
            vectors_type18.append(vector)
            i+=1
        return vectors_type18

    #vector that uses the pos of the word before and after the smixut
    def make_vector19(self):
        # Create a list to store the resulting vectors
        vectors_pos_tags = []

        for i, smixut in enumerate(self.smixut_list):
            if i >= len(self.sentences):
                break


            word1, word2 = smixut.split(' ', 1)
            vector = [0] * len(self.unique_pos)

            # Find the POS tag of the word before the smixut
            word_before = self.find_dict_in_shift_from_smixut(self.sentences[i], -1, word1)
            if word_before != "err":
                pos_before = word_before.get("pos")
                if pos_before in self.unique_pos:
                    vector[self.unique_pos[pos_before]] = 1

            # Find the POS tag of the word after the smixut
            word_after = self.find_dict_in_shift_from_smixut(self.sentences[i], 1, word2)
            if word_after != "err":
                pos_after = word_after.get("pos")
                if pos_after in self.unique_pos:
                    vector[self.unique_pos[pos_after]] = 1

            # Append the vector to the list of vectors
            vectors_pos_tags.append(vector)

        return vectors_pos_tags

    #vector that uses the pos tags and lex of the words after and befroe smixut
    def make_vector20(self):
        # Create a list to store the resulting vectors
        combined_vectors = []
        for i, smixut in enumerate(self.smixut_list):
            if i >= len(self.sentences):
                break

            word1, word2 = smixut.split(' ', 1)

            # Initialize the vector with zeros
            vector = [0] * (len(self.unique_pos) + len(self.unique_words_lex))

            # Process the word before the smixut
            word_before = self.find_dict_in_shift_from_smixut(self.sentences[i], -1, word1)
            if word_before != "err":
                pos_before = word_before.get("pos")
                lex_before = word_before.get("lex")
                if pos_before in self.unique_pos:
                    vector[self.unique_pos[pos_before]] = 1
                if lex_before in self.unique_words_lex:
                    vector[len(self.unique_pos) + self.unique_words_lex[lex_before]] = 1

            # Process the word after the smixut
            word_after = self.find_dict_in_shift_from_smixut(self.sentences[i], 1, word2)
            if word_after != "err":
                pos_after = word_after.get("pos")
                lex_after = word_after.get("lex")
                if pos_after in self.unique_pos:
                    vector[self.unique_pos[pos_after]] = 1
                if lex_after in self.unique_words_lex:
                    vector[len(self.unique_pos) + self.unique_words_lex[lex_after]] = 1

            # Append the vector to the list of vectors
            combined_vectors.append(vector)

        return combined_vectors

    #vector that uses the pos tags and unqiue of the 2 words before
    def make_vector21(self):
        # Create a list to store the resulting vectors
        combined_vectors = []

        for i, smixut in enumerate(self.smixut_list):
            if i >= len(self.sentences):
                break

            word1, word2 = smixut.split(' ', 1)

            # Initialize the vector with zeros
            vector = [0] * (len(self.unique_pos) + len(self.unique_words))

            # Process the word before the smixut
            word_before = self.find_dict_in_shift_from_smixut(self.sentences[i], -1, word1)
            if word_before != "err":
                pos_before = word_before.get("pos")
                word_before_text = word_before.get("word")
                if pos_before in self.unique_pos:
                    vector[self.unique_pos[pos_before]] = 1
                if word_before_text in self.unique_words:
                    vector[len(self.unique_pos) + self.unique_words[word_before_text]] = 1

            # Process the word after the smixut
            word_before_2 = self.find_dict_in_shift_from_smixut(self.sentences[i], -2, word2)
            if word_before_2  != "err":
                pos_after = word_before_2 .get("pos")
                word_before2_text = word_before_2 .get("word")
                if pos_after in self.unique_pos:
                    vector[self.unique_pos[pos_after]] = 1
                if word_before2_text in self.unique_words:
                    vector[len(self.unique_pos) + self.unique_words[word_before2_text]] = 1

            # Append the vector to the list of vectors
            combined_vectors.append(vector)

        return combined_vectors

    #vector that uses the pos tags and lex of the words after and befroe smixut
    def make_vector22(self):
        # Create a list to store the resulting vectors
        combined_vectors = []

        for i, smixut in enumerate(self.smixut_list):
            if i >= len(self.sentences):
                break

            word1, word2 = smixut.split(' ', 1)

            # Initialize the vector with zeros
            vector = [0] * (len(self.unique_pos) + len(self.unique_words_lex))

            # Process the word before the smixut
            word_before = self.find_dict_in_shift_from_smixut(self.sentences[i], -1, word1)
            if word_before != "err":
                pos_before = word_before.get("pos")
                lex_before = word_before.get("lex")
                if pos_before in self.unique_pos:
                    vector[self.unique_pos[pos_before]] = 1
                if lex_before in self.unique_words_lex:
                    vector[len(self.unique_pos) + self.unique_words_lex[lex_before]] = 1

            # Process the word after the smixut
            word_after = self.find_dict_in_shift_from_smixut(self.sentences[i], 1, word2)
            if word_after != "err":
                pos_after = word_after.get("pos")
                lex_after = word_after.get("lex")
                if pos_after in self.unique_pos:
                    vector[self.unique_pos[pos_after]] = 1
                if lex_after in self.unique_words_lex:
                    vector[len(self.unique_pos) + self.unique_words_lex[lex_after]] = 1

            # Append the vector to the list of vectors
            combined_vectors.append(vector)

        return combined_vectors

    #vector that uses the pos of the two words before smixut
    def make_vector23(self):
        # Create a list to store the resulting vectors
        vectors_pos_tags = []

        for i, smixut in enumerate(self.smixut_list):
            if i >= len(self.sentences):
                break


            word1, word2 = smixut.split(' ', 1)
            vector = [0] * len(self.unique_pos)

            # Find the POS tag of the word before the smixut
            word_before = self.find_dict_in_shift_from_smixut(self.sentences[i], -1, word1)
            if word_before != "err":
                pos_before = word_before.get("pos")
                if pos_before in self.unique_pos:
                    vector[self.unique_pos[pos_before]] = 1

            # Find the POS tag of the word after the smixut
            word_before2 = self.find_dict_in_shift_from_smixut(self.sentences[i], -2, word2)
            if word_before2 != "err":
                pos_before2 = word_before2.get("pos")
                if pos_before2 in self.unique_pos:
                    vector[self.unique_pos[pos_before2]] = 1

            # Append the vector to the list of vectors
            vectors_pos_tags.append(vector)

        return vectors_pos_tags

    #vector that uses the pos tags and lex of the two words before smixut
    def make_vector24(self):
        # Create a list to store the resulting vectors
        combined_vectors = []
        for i, smixut in enumerate(self.smixut_list):
            if i >= len(self.sentences):
                break

            word1, word2 = smixut.split(' ', 1)

            # Initialize the vector with zeros
            vector = [0] * (len(self.unique_pos) + len(self.unique_words_lex))

            # Process the word before the smixut
            word_before = self.find_dict_in_shift_from_smixut(self.sentences[i], -1, word1)
            if word_before != "err":
                pos_before = word_before.get("pos")
                lex_before = word_before.get("lex")
                if pos_before in self.unique_pos:
                    vector[self.unique_pos[pos_before]] = 1
                if lex_before in self.unique_words_lex:
                    vector[len(self.unique_pos) + self.unique_words_lex[lex_before]] = 1

            # Process the word after the smixut
            word_before2 = self.find_dict_in_shift_from_smixut(self.sentences[i], -2, word2)
            if word_before2  != "err":
                pos_before2 = word_before2 .get("pos")
                lex_before2 = word_before2 .get("lex")
                if pos_before2 in self.unique_pos:
                    vector[self.unique_pos[pos_before2]] = 1
                if lex_before2 in self.unique_words_lex:
                    vector[len(self.unique_pos) + self.unique_words_lex[lex_before2]] = 1

            # Append the vector to the list of vectors
            combined_vectors.append(vector)
        return combined_vectors

    #vector that uses the dep funcs and lex of the two words one before one after smixut
    def make_vector25(self):
        # Create a list to store the resulting vectors
        dep_lex_vectors = []

        for i, smixut in enumerate(self.smixut_list):
            if i >= len(self.sentences):
                break

            word1, word2 = smixut.split(' ', 1)

            # Initialize the vector with zeros
            vector = [0] * (len(self.unique_dep_func) + len(self.unique_words_lex))

            # Process the word before the smixut
            word_before = self.find_dict_in_shift_from_smixut(self.sentences[i], -1, word1)
            if word_before != "err":
                dep_func_before = word_before.get("dep_func")
                lex_before = word_before.get("lex")
                if dep_func_before in self.unique_dep_func:
                    vector[self.unique_dep_func[dep_func_before]] = 1
                if lex_before in self.unique_words_lex:
                    vector[len(self.unique_dep_func) + self.unique_words_lex[lex_before]] = 1

            # Process the word after the smixut
            word_after = self.find_dict_in_shift_from_smixut(self.sentences[i], 1, word2)
            if word_after != "err":
                dep_func_after = word_after.get("dep_func")
                lex_after = word_after.get("lex")
                if dep_func_after in self.unique_dep_func:
                    vector[self.unique_dep_func[dep_func_after]] = 1
                if lex_after in self.unique_words_lex:
                    vector[len(self.unique_dep_func) + self.unique_words_lex[lex_after]] = 1

            # Append the vector to the list of vectors
            dep_lex_vectors.append(vector)

        return dep_lex_vectors

    #vector that uses the dep funcs and lex of the two words before smixut
    def make_vector26(self):
        # Create a list to store the resulting vectors
        dep_lex_vectors = []

        for i, smixut in enumerate(self.smixut_list):
            if i >= len(self.sentences):
                break

            word1, word2 = smixut.split(' ', 1)

            # Initialize the vector with zeros
            vector = [0] * (len(self.unique_dep_func) + len(self.unique_words_lex))

            # Process the word before the smixut
            word_before = self.find_dict_in_shift_from_smixut(self.sentences[i], -1, word1)
            if word_before != "err":
                dep_func_before = word_before.get("dep_func")
                lex_before = word_before.get("lex")
                if dep_func_before in self.unique_dep_func:
                    vector[self.unique_dep_func[dep_func_before]] = 1
                if lex_before in self.unique_words_lex:
                    vector[len(self.unique_dep_func) + self.unique_words_lex[lex_before]] = 1

            # Process the word after the smixut
            word_after = self.find_dict_in_shift_from_smixut(self.sentences[i], -2, word2)
            if word_after != "err":
                dep_func_after = word_after.get("dep_func")
                lex_after = word_after.get("lex")
                if dep_func_after in self.unique_dep_func:
                    vector[self.unique_dep_func[dep_func_after]] = 1
                if lex_after in self.unique_words_lex:
                    vector[len(self.unique_dep_func) + self.unique_words_lex[lex_after]] = 1

            # Append the vector to the list of vectors
            dep_lex_vectors.append(vector)

        return dep_lex_vectors

    #vector that uses the dep funcs of word after and before of smixut
    def make_vector27(self):
        # Create a list to store the resulting vectors
        dep_func_vectors = []

        for i, smixut in enumerate(self.smixut_list):
            if i >= len(self.sentences):
                break

            word1, word2 = smixut.split(' ', 1)

            # Initialize the vector with zeros
            vector = [0] * (2 * len(self.unique_dep_func))

            # Process the word before the smixut
            word_before = self.find_dict_in_shift_from_smixut(self.sentences[i], -1, word1)
            if word_before != "err":
                dep_func_before = word_before.get("dep_func")
                if dep_func_before in self.unique_dep_func:
                    vector[self.unique_dep_func[dep_func_before]] = 1

            # Process the word after the smixut
            word_after = self.find_dict_in_shift_from_smixut(self.sentences[i], 1, word2)
            if word_after != "err":
                dep_func_after = word_after.get("dep_func")
                if dep_func_after in self.unique_dep_func:
                    vector[len(self.unique_dep_func) + self.unique_dep_func[dep_func_after]] = 1

            # Append the vector to the list of vectors
            dep_func_vectors.append(vector)

        return dep_func_vectors

    #vector that uses the dep funcs and unique words of the two words one before one after smixut
    def make_vector28(self):
        # Create a list to store the resulting vectors
        dep_func_word_vectors = []

        for i, smixut in enumerate(self.smixut_list):
            if i >= len(self.sentences):
                break

            word1, word2 = smixut.split(' ', 1)

            # Initialize the vector with zeros
            vector = [0] * (2 * len(self.unique_dep_func) + 2 * len(self.unique_words))

            # Process the word before the smixut
            word_before = self.find_dict_in_shift_from_smixut(self.sentences[i], -1, word1)
            if word_before != "err":
                dep_func_before = word_before.get("dep_func")
                word_before_text = word_before.get("word")
                if dep_func_before in self.unique_dep_func:
                    vector[self.unique_dep_func[dep_func_before]] = 1
                if word_before_text in self.unique_words:
                    vector[len(self.unique_dep_func) + self.unique_words[word_before_text]] = 1

            # Process the word after the smixut
            word_after = self.find_dict_in_shift_from_smixut(self.sentences[i], 1, word2)
            if word_after != "err":
                dep_func_after = word_after.get("dep_func")
                word_after_text = word_after.get("word")
                if dep_func_after in self.unique_dep_func:
                    vector[len(self.unique_dep_func) * 2 + self.unique_dep_func[dep_func_after]] = 1
                if word_after_text in self.unique_words:
                    vector[len(self.unique_dep_func) * 2 + len(self.unique_words) + self.unique_words[word_after_text]] = 1

            # Append the vector to the list of vectors
            dep_func_word_vectors.append(vector)

        return dep_func_word_vectors

    #vector that uses the dep funcs and unique words of the two words before smixut
    def make_vector29(self):
        # Create a list to store the resulting vectors
        dep_func_word_vectors = []

        for i, smixut in enumerate(self.smixut_list):
            if i >= len(self.sentences):
                break

            word1, word2 = smixut.split(' ', 1)

            # Initialize the vector with zeros
            vector = [0] * (2 * len(self.unique_dep_func) + 2 * len(self.unique_words))

            # Process the word before the smixut
            word_before = self.find_dict_in_shift_from_smixut(self.sentences[i], -1, word1)
            if word_before != "err":
                dep_func_before = word_before.get("dep_func")
                word_before_text = word_before.get("word")
                if dep_func_before in self.unique_dep_func:
                    vector[self.unique_dep_func[dep_func_before]] = 1
                if word_before_text in self.unique_words:
                    vector[len(self.unique_dep_func) + self.unique_words[word_before_text]] = 1

            # Process the word after the smixut
            word_after = self.find_dict_in_shift_from_smixut(self.sentences[i], -2, word2)
            if word_after != "err":
                dep_func_after = word_after.get("dep_func")
                word_after_text = word_after.get("word")
                if dep_func_after in self.unique_dep_func:
                    vector[len(self.unique_dep_func) * 2 + self.unique_dep_func[dep_func_after]] = 1
                if word_after_text in self.unique_words:
                    vector[len(self.unique_dep_func) * 2 + len(self.unique_words) + self.unique_words[word_after_text]] = 1

            # Append the vector to the list of vectors
            dep_func_word_vectors.append(vector)

        return dep_func_word_vectors

    #vector that uses the dep funcs of two words before smixut
    def make_vector30(self):
        # Create a list to store the resulting vectors
        dep_func_vectors = []
        for i, smixut in enumerate(self.smixut_list):
            if i >= len(self.sentences):
                break

            word1, word2 = smixut.split(' ', 1)

            # Initialize the vector with zeros
            vector = [0] * (2 * len(self.unique_dep_func))

            # Process the word before the smixut
            word_before = self.find_dict_in_shift_from_smixut(self.sentences[i], -1, word1)
            if word_before != "err":
                dep_func_before = word_before.get("dep_func")
                if dep_func_before in self.unique_dep_func:
                    vector[self.unique_dep_func[dep_func_before]] = 1

            # Process the word after the smixut
            word_after = self.find_dict_in_shift_from_smixut(self.sentences[i], -2, word2)
            if word_after != "err":
                dep_func_after = word_after.get("dep_func")
                if dep_func_after in self.unique_dep_func:
                    vector[len(self.unique_dep_func) + self.unique_dep_func[dep_func_after]] = 1

            # Append the vector to the list of vectors
            dep_func_vectors.append(vector)

        return dep_func_vectors



    
