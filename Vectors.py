
import json
import pandas as pd
import string
from transformers import AutoModel, AutoTokenizer
from initializer import DataStore

tokenizer = AutoTokenizer.from_pretrained('dicta-il/dictabert-tiny-joint')
model = AutoModel.from_pretrained('dicta-il/dictabert-tiny-joint', trust_remote_code=True)

model.eval()
data = DataStore()
sentences = data.sentences
smixut_list = data.smixut_list
unique_words = data.unique_words
unique_genders = data.unique_genders
unique_pos = data.unique_pos
unique_dep_func = data.unique_dep_func
unique_words_lex = data.unique_words_lex

def find_token_info(data, search_token):
    for entry in data:
        for token_info in entry['tokens']:
            if token_info['token'] == search_token:
                return token_info
    return None


def print_to_file(token_info, file):
    if token_info:
        # with open('output.txt', 'w', encoding='utf-8') as file:
        file.write(json.dumps(token_info, ensure_ascii=False, indent=4))
    else:
        print(f"Token not found.")


# def convert_json_to_dict(json_data):
#     # Extract the necessary fields from the JSON data
#     word = json_data.get("token")
#     lex = json_data.get("lex")
#     pos = json_data.get("morph", {}).get("pos")
#     gender = json_data.get("morph", {}).get("feats", {}).get("Gender")
#     dep_func = json_data.get("syntax", {}).get("dep_func")
#
#     # Create the dictionary with the desired keys
#     result = {
#         "word": word,
#         "lex": lex,
#         "pos": pos,
#         "Gender": gender,
#         "dep_func": dep_func
#     }
#
#     return result


# def process_json(input_json):
#     # Extract tokens from the input JSON
#     tokens = input_json[0].get("tokens", [])
#
#     # Convert each token to the desired dictionary format
#     result = [convert_json_to_dict(token) for token in tokens]
#
#     return result
#
#
# def add_unique_words_lex_to_dict(sentence_list, word_dict):
#     punctuation = r"\"#$%&'()*+,-–./:;<=>?@[\]^_`{|}~"
#     for item in sentence_list:
#         lex_word = item.get("lex")
#         if not (len(lex_word) == 1 and lex_word in punctuation) and (lex_word and lex_word not in word_dict):
#             word_dict[lex_word] = len(word_dict)
#     return word_dict
#
# def add_unique_words_to_dict(sentence_list, word_dict):
#     punctuation = r"\"#$%&'()*+,-–./:;<=>?@[\]^_`{|}~"
#     for item in sentence_list:
#         lex_word = item.get("word")
#         if not (lex_word in punctuation) and (lex_word and lex_word not in word_dict):
#             word_dict[lex_word] = len(word_dict)
#     return word_dict
#
# def add_unique_pos_to_dict(sentence_list, pos_dict):
#     punctuation = r"\"#$%&'()*+,-–./:;<=>?@[\]^_`{|}~"
#     for item in sentence_list:
#         lex_word = item.get("word")
#         lex_pos = item.get("pos")
#         if not (lex_word in punctuation) and (lex_pos and lex_pos not in pos_dict):
#             pos_dict[lex_pos] = len(pos_dict)
#     return pos_dict
#
# def add_unique_dep_func_to_dict(sentence_list, dep_func_dict):
#     punctuation = r"\"#$%&'()*+,-–./:;<=>?@[\]^_`{|}~"
#     for item in sentence_list:
#         lex_word = item.get("word")
#         lex_dep_func = item.get("dep_func")
#         if not (lex_word in punctuation) and (lex_dep_func and lex_dep_func not in dep_func_dict):
#             dep_func_dict[lex_dep_func] = len(dep_func_dict)
#     return dep_func_dict

def get_lex(word,sentence):
    for item in sentence:
        if item.get("word") == word:
            return item.get("lex")
    return "error"

def find_string_in_dict_list(sentence, target):
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
def find_dict_in_shift_from_smixut(sentence, shift, smixut):
    i=0
    punctuation = r"\"#$%&'()*+,-–./:;<=>?@[\]^_`{|}~"
    for word_dict in sentence:
        if (smixut in word_dict.get("word")) and (0 < i+shift < len(sentence)):
            if(sentence[i+shift].get("word") in punctuation):
                if(shift > 0):
                    return find_dict_in_shift_from_smixut(sentence, shift+1, smixut)
                else:
                    return find_dict_in_shift_from_smixut(sentence, shift-1, smixut)
            else:
                return sentence[i+shift]
        i+=1
    return "err"


def find_pos_in_dict_list(sentence, target):
    for word_dict in sentence:
        if target in word_dict.get("word"):
            return word_dict.get("pos")
    return "err"  # Return -1 if the target string is not found

def find_dep_in_dict_list(sentence, target):
    for word_dict in sentence:
        if target in word_dict.get("word"):
            return word_dict.get("dep_func")
    return "err"  # Return -1 if the target string is not found


# def add_smixut_to_uniqe_words(smixut_list, uniqe_words):
#     punctuation = r"\"#$%&'()*+,-–./:;<=>?@[\]^_`{|}~"
#     for smixut in smixut_list:
#         if '-' not in smixut or '–' not in smixut:
#             word1, word2 = smixut.split(' ', 1)
#             if not (len(word1) == 1 and word1 in punctuation) and (word1 and word1 not in uniqe_words):
#                 uniqe_words[word1] = len(uniqe_words)
#             if not (len(word2) == 1 and word2 in punctuation) and (word2 and word2 not in uniqe_words):
#                 uniqe_words[word2] = len(uniqe_words)
#
#     return uniqe_words


   # Returns the key associated with the given value in a dictionary.

   # Args:
     #   d (dict): The dictionary to search.
      #  target_value: The value to find the key for.

   # Returns:
     #   The key associated with the target_value if found, else None.

def get_key_by_value(dict, target_value):
    for key, value in dict.items():
        if key == target_value:
            return value
    return None


def find_smixut_location_in_sentence(smixut, sentence):
    word1, word2 = smixut.split(' ', 1)
    ind = 0
    for word_dict in sentence:
        if word1 == word_dict.get("word"):
            break
        ind += 1
    return ind

def get_unique_pos():
    # Create a dictionary of unique POS tags with their corresponding index
    unique_pos_tags = {
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
    return unique_pos_tags

def get_unique_dep_func():
    unique_dep_funcs = {
        "nsubj": 0,  # Nominal Subject
        "obj": 1,  # Object
        "obl": 2,  # Oblique
        "advmod": 3,  # Adverbial Modifier
        "acl:relcl": 4,  # Relative Clause Modifier
        "compound:smixut": 5,  # Compound Smixut
        "cop": 6,  # Copula
        "punct": 7  # Punctuation
    }

    return unique_dep_funcs

#vector that uses the word of the smixut
def make_vector1(smixut_list, unique_words, sentences):
    #create list of vectors
    vectors_type1 = []

    for smixut in smixut_list:
        word1, word2 = smixut.split(' ', 1)
        # find_token_info()
        vector = [0] * len(unique_words)
        # complete get_lex
        vector[unique_words[word1]] = 1
        vector[unique_words[word2]] = 1
        vectors_type1.append(vector)
    return vectors_type1

#vector that uses the lex of the smixut
def make_vector2(smixut_list, unique_lex_words,sentences):
    #create list of vectors
    vectors_type2 = []
    i = 0
    for smixut in smixut_list:
        if(i >= len(sentences)):
            break
        word1, word2 = smixut.split(' ', 1)
        lex_word1 = find_string_in_dict_list(sentences[i], word1)
        lex_word2 = find_string_in_dict_list(sentences[i], word2)
        # find_token_info()
        vector = [0] * len(unique_lex_words)
        # complete get_lex
        if(lex_word1 != "err"):
            vector[unique_lex_words[lex_word1]] = 1
        if (lex_word2 != "err"):
            vector[unique_lex_words[lex_word2]] = 1
        vectors_type2.append(vector)
        i += 1

    return vectors_type2

#vector that marks all the words in the sentence besides the smixut
#assuming that the smixut is the word where "dep_func": "compound:smixut" and the previous one
def make_vector3(unique_words, smixut_list, sentences):
    vectors_type3 = []
    punctuation = r"\"#$%&'()*+,-–./:;<=>?@[\]^_`{|}~"
    for i, sentence in enumerate(sentences):
        if i >= len(smixut_list):
            break
        curr_vec = [0] * len(unique_words)
        word1, word2 = smixut_list[i].split(' ', 1)
        for word_dict in sentence:
            curr_word = word_dict.get("word")
            if curr_word not in punctuation:
                if not (curr_word == word1 or curr_word == word2):
                    curr_vec[unique_words[curr_word]] = 1
        vectors_type3.append(curr_vec)
    return vectors_type3

#vector that uses the pos of the smixut
def make_vector4(smixut_list, unique_pos,sentences):
    vector_type4 = []
    i=0
    for smixut in smixut_list:
        if(i >= len(sentences)):
            break
        word1, word2 = smixut.split(' ', 1)
        pos_word1 = find_pos_in_dict_list(sentences[i], word1)
        pos_word2 = find_pos_in_dict_list(sentences[i], word2)
        # find_token_info()
        vector = [0] * len(unique_pos)
        # complete get_lex
        if(pos_word1 != "err"):
            vector[unique_pos[pos_word1]] = 1
        if (pos_word2 != "err"):
            vector[unique_pos[pos_word2]] = 1
        vector_type4.append(vector)
        i += 1
    return vector_type4

#vector that uses the dep_func of the smixut
def make_vector5(smixut_list, unique_dep_func,sentences):
    vector_type5 = []
    i=0
    for smixut in smixut_list:
        if(i >= len(sentences)):
            break
        word1, word2 = smixut.split(' ', 1)
        dep_word1 = find_dep_in_dict_list(sentences[i], word1)
        dep_word2 = find_dep_in_dict_list(sentences[i], word2)
        # find_token_info()
        vector = [0] * len(unique_dep_func)
        # complete get_lex
        if(dep_word1 != "err"):
            vector[unique_dep_func[dep_word1]] = 1
        if (dep_word2 != "err"):
            vector[unique_dep_func[dep_word2]] = 1
        vector_type5.append(vector)
        i += 1
    return vector_type5

#vector that uses the word of the one word before and after the smixut
def make_vector6(smixut_list, unique_words, sentences):
    #create list of vectors
    vectors_type6 = []
    i = 0
    for smixut in smixut_list:
        word1, word2 = smixut.split(' ', 1)
        # find_token_info()
        if (i >= len(sentences)):
            break
        vector = [0] * len(unique_words)
        word_before = find_dict_in_shift_from_smixut(sentences[i], -1, word1)
        word_after = find_dict_in_shift_from_smixut(sentences[i], 1, word2)
        if((word_before != "err")  ):
            vector[unique_words[word_before.get("word")]] = 1
        if ((word_after != "err")  ):
            vector[unique_words[word_after.get("word")]] = 1
        vectors_type6.append(vector)
        i+=1
    return vectors_type6

#vector that uses the word before and after smixut with pos and dep func
def make_vector7(smixut_list, sentences):
    # Create a list to store the resulting vectors
    pos_dep_func_vectors = []
    unique_dep_funcs= get_unique_dep_func()
    unique_pos_tags = get_unique_pos()

    for i, smixut in enumerate(smixut_list):
        if i >= len(sentences):
            break

        word1, word2 = smixut.split(' ', 1)

        # Initialize the vector with zeros
        vector = [0] * (2 * len(unique_pos_tags) + 2 * len(unique_dep_funcs))

        # Process the word before the smixut
        word_before = find_dict_in_shift_from_smixut(sentences[i], -1, word1)
        if word_before != "err":
            pos_before = word_before.get("pos")
            dep_func_before = word_before.get("dep_func")
            if pos_before in unique_pos_tags:
                vector[unique_pos_tags[pos_before]] = 1
            if dep_func_before in unique_dep_funcs:
                vector[len(unique_pos_tags) + unique_dep_funcs[dep_func_before]] = 1

        # Process the word after the smixut
        word_after = find_dict_in_shift_from_smixut(sentences[i], 1, word2)
        if word_after != "err":
            pos_after = word_after.get("pos")
            dep_func_after = word_after.get("dep_func")
            if pos_after in unique_pos_tags:
                vector[len(unique_pos_tags) * 2 + unique_pos_tags[pos_after]] = 1
            if dep_func_after in unique_dep_funcs:
                vector[len(unique_pos_tags) * 2 + len(unique_dep_funcs) + unique_dep_funcs[dep_func_after]] = 1

        # Append the vector to the list of vectors
        pos_dep_func_vectors.append(vector)

    return pos_dep_func_vectors

#vector that uses the word of 2 words before and after the smixut
def make_vector8(smixut_list, unique_words, sentences):
    punctuation = r"\"#$%&'()*+,-–./:;<=>?@[\]^_`{|}~"
    #create list of vectors
    vectors_type8 = []
    i = 0
    for smixut in smixut_list:
        word1, word2 = smixut.split(' ', 1)
        # find_token_info()
        if (i >= len(sentences)):
            break
        vector = [0] * len(unique_words)
        one_word_before = find_dict_in_shift_from_smixut(sentences[i], -1, word1)
        two_words_before = find_dict_in_shift_from_smixut(sentences[i], -2, word1)
        one_word_after = find_dict_in_shift_from_smixut(sentences[i], 1, word2)
        two_words_after = find_dict_in_shift_from_smixut(sentences[i], 2, word2)
        if ((one_word_before != "err") and (not (one_word_before.get("word") in punctuation))):
            vector[unique_words[one_word_before.get("word")]] = 1
        if ((two_words_before != "err") and (not (two_words_before.get("word") in punctuation)) ):
            vector[unique_words[two_words_before.get("word")]] = 1
        if ((one_word_after != "err") and (not (one_word_after.get("word") in punctuation))):
            vector[unique_words[one_word_after.get("word")]] = 1
        if ((two_words_after != "err") and (not (two_words_after.get("word") in punctuation))):
            vector[unique_words[two_words_after.get("word")]] = 1

        vectors_type8.append(vector)
        i += 1
    return vectors_type8

# vector that for each word in the smixut puts the number that represents its dep_func
def make_vector9(smixut_list, unique_words, sentences, unique_dep_func):
    #create list of vectors
    vectors_type9 = []
    i=0
    for smixut in smixut_list:
        if (i >= len(sentences)):
            break
        word1, word2 = smixut.split(' ', 1)
        # find_token_info()
        vector = [0] * len(unique_words)
        dict1 = find_dict_in_shift_from_smixut(sentences[i], 0, word1)
        dict2 = find_dict_in_shift_from_smixut(sentences[i], 0, word2)
        # target_value is the index of word1's dep_func in unique_dep_func
        if dict1 != "err":
            vector[unique_words[word1]] = get_key_by_value(unique_dep_func, dict1.get("dep_func"))
        if dict2 != "err":
            vector[unique_words[word2]] = get_key_by_value(unique_dep_func, dict2.get("dep_func"))
        vectors_type9.append(vector)
        i+=1
    return vectors_type9

#vector that uses the lex of the one word before and after the smixut
def make_vector10(smixut_list, unique_lex_words ,sentences):
    #create list of vectors
    vectors_type10 = []
    i = 0
    for smixut in smixut_list:
        if (i >= len(sentences)):
            break
        word1, word2 = smixut.split(' ', 1)
        # find_token_info()
        if (i >= len(sentences)):
            break
        vector = [0] * len(unique_words)
        word_before = find_dict_in_shift_from_smixut(sentences[i], -1, word1)
        word_after = find_dict_in_shift_from_smixut(sentences[i], 1, word2)
        if((word_before != "err")  ):
            vector[unique_words[word_before.get("lex")]] = 1
        if ((word_after != "err")  ):
            vector[unique_words[word_after.get("lex")]] = 1
        vectors_type10.append(vector)
        i+=1
    return vectors_type10
# vector that for each word in the smixut puts the number that represents its lex
def make_vector11(smixut_list, unique_words, sentences, unique_lex):
    #create list of vectors
    vectors_type11 = []
    i=0
    for smixut in smixut_list:
        if (i >= len(sentences)):
            break
        word1, word2 = smixut.split(' ', 1)
        # find_token_info()
        vector = [0] * len(unique_words)
        dict1 = find_dict_in_shift_from_smixut(sentences[i], 0, word1)
        dict2 = find_dict_in_shift_from_smixut(sentences[i], 0, word2)
        # target_value is the index of word1's dep_func in unique_dep_func
        if dict1 != "err":
            vector[unique_words[word1]] = get_key_by_value(unique_lex, dict1.get("lex"))
        if dict2 != "err":
            vector[unique_words[word2]] = get_key_by_value(unique_lex, dict2.get("lex"))
        vectors_type11.append(vector)
        i+=1
    return vectors_type11

#vector that for one word before and one after the smixut puts the number that represents its lex
def make_vector12(smixut_list, unique_words, sentences, unique_lex):
    #create list of vectors
    vectors_type12 = []
    i=0
    for smixut in smixut_list:
        if (i >= len(sentences)):
            break
        word1, word2 = smixut.split(' ', 1)
        # find_token_info()
        vector = [0] * len(unique_words)
        dict1 = find_dict_in_shift_from_smixut(sentences[i], -1, word1)
        dict2 = find_dict_in_shift_from_smixut(sentences[i], 1, word2)
        # target_value is the index of word1's dep_func in unique_dep_func
        if dict1 != "err":
            vector[unique_words[dict1.get("word")]] = get_key_by_value(unique_lex, dict1.get("lex"))
        if dict2 != "err":
            vector[unique_words[dict2.get("word")]] = get_key_by_value(unique_lex, dict2.get("lex"))
        vectors_type12.append(vector)
        i+=1
    return vectors_type12


#vector that for one word before and one after the smixut takes its pos puts the number that represents its lex
def make_vector13(smixut_list, unique_pos, sentences, unique_lex):
    #create list of vectors
    vectors_type13 = []
    i=0
    for smixut in smixut_list:
        if (i >= len(sentences)):
            break
        word1, word2 = smixut.split(' ', 1)
        # find_token_info()
        vector = [0] * len(unique_pos)
        dict1 = find_dict_in_shift_from_smixut(sentences[i], -1, word1)
        dict2 = find_dict_in_shift_from_smixut(sentences[i], 1, word2)
        if dict1 != "err":
            vector[unique_pos[dict1.get("pos")]] = get_key_by_value(unique_lex, dict1.get("lex"))
        if dict2 != "err":
            vector[unique_pos[dict2.get("pos")]] = get_key_by_value(unique_lex, dict2.get("lex"))
        vectors_type13.append(vector)
        i+=1
    return vectors_type13

#vector that uses the word of the two words after the smixut
def make_vector14(smixut_list, unique_words, sentences):
    #create list of vectors
    vectors_type14 = []
    i = 0
    for smixut in smixut_list:
        word1, word2 = smixut.split(' ', 1)
        # find_token_info()
        if (i >= len(sentences)):
            break
        vector = [0] * len(unique_words)
        word_after = find_dict_in_shift_from_smixut(sentences[i], 1, word2)
        word_after2 = find_dict_in_shift_from_smixut(sentences[i], 2, word2)
        if((word_after != "err")  ):
            vector[unique_words[word_after.get("word")]] = 1
        if ((word_after2 != "err")  ):
            vector[unique_words[word_after2.get("word")]] = 1
        vectors_type14.append(vector)
        i+=1
    return vectors_type14


#vector that uses the lex of the two words after the smixut
def make_vector15(smixut_list, unique_lex_words, sentences):
    #create list of vectors
    vectors_type15 = []
    i = 0
    for smixut in smixut_list:
        word1, word2 = smixut.split(' ', 1)
        # find_token_info()
        if (i >= len(sentences)):
            break
        vector = [0] * len(unique_lex_words)
        word_after = find_dict_in_shift_from_smixut(sentences[i], 1, word2)
        word_after2 = find_dict_in_shift_from_smixut(sentences[i], 2, word2)
        if ((word_after != "err")):
            vector[unique_lex_words[word_after.get("lex")]] = 1
        if ((word_after2 != "err")):
            vector[unique_lex_words[word_after2.get("lex")]] = 1
        vectors_type15.append(vector)
        i+=1
    return vectors_type15

#vector that for two words after the smixut puts the number that represents its lex in the word part vector

def make_vector16(smixut_list, unique_words, sentences, unique_lex):
    #create list of vectors
    vectors_type16 = []
    i=0
    for smixut in smixut_list:
        if (i >= len(sentences)):
            break
        word1, word2 = smixut.split(' ', 1)
        # find_token_info()
        vector = [0] * len(unique_words)
        dict1 = find_dict_in_shift_from_smixut(sentences[i], 1, word2)
        dict2 = find_dict_in_shift_from_smixut(sentences[i], 2, word2)
        # target_value is the index of word1's dep_func in unique_dep_func
        if dict1 != "err":
            vector[unique_words[dict1.get("word")]] = get_key_by_value(unique_lex, dict1.get("lex"))
        if dict2 != "err":
            vector[unique_words[dict2.get("word")]] = get_key_by_value(unique_lex, dict2.get("lex"))
        vectors_type16.append(vector)
        i+=1
    return vectors_type16

#vector that uses the word of the two words before the smixut
def make_vector17(smixut_list, unique_words, sentences):
    #create list of vectors
    vectors_type17 = []
    i = 0
    for smixut in smixut_list:
        word1, word2 = smixut.split(' ', 1)
        # find_token_info()
        if (i >= len(sentences)):
            break
        vector = [0] * len(unique_words)
        word_before = find_dict_in_shift_from_smixut(sentences[i], -1, word1)
        word_before2 = find_dict_in_shift_from_smixut(sentences[i], -2, word1)
        if((word_before != "err")  ):
            vector[unique_words[word_before.get("word")]] = 1
        if ((word_before2 != "err")  ):
            vector[unique_words[word_before2.get("word")]] = 1
        vectors_type17.append(vector)
        i+=1
    return vectors_type17


#vector that uses the lex of the two words before the smixut
def make_vector18(smixut_list, unique_lex_words, sentences):
    #create list of vectors
    vectors_type18 = []
    i = 0
    for smixut in smixut_list:
        word1, word2 = smixut.split(' ', 1)
        # find_token_info()
        if (i >= len(sentences)):
            break
        vector = [0] * len(unique_lex_words)
        word_before = find_dict_in_shift_from_smixut(sentences[i], -1, word1)
        word_before2 = find_dict_in_shift_from_smixut(sentences[i], -2, word1)
        if ((word_before != "err")):
            vector[unique_lex_words[word_before.get("lex")]] = 1
        if ((word_before2 != "err")):
            vector[unique_lex_words[word_before2.get("lex")]] = 1
        vectors_type18.append(vector)
        i+=1
    return vectors_type18

#vector that uses the pos of the word before and after the smixut
def make_vector19(smixut_list, sentences):
    # Create a list to store the resulting vectors
    vectors_pos_tags = []

    unique_pos_tags = get_unique_pos()

    for i, smixut in enumerate(smixut_list):
        if i >= len(sentences):
            break


        word1, word2 = smixut.split(' ', 1)
        vector = [0] * len(unique_pos_tags)

        # Find the POS tag of the word before the smixut
        word_before = find_dict_in_shift_from_smixut(sentences[i], -1, word1)
        if word_before != "err":
            pos_before = word_before.get("pos")
            if pos_before in unique_pos_tags:
                vector[unique_pos_tags[pos_before]] = 1

        # Find the POS tag of the word after the smixut
        word_after = find_dict_in_shift_from_smixut(sentences[i], 1, word2)
        if word_after != "err":
            pos_after = word_after.get("pos")
            if pos_after in unique_pos_tags:
                vector[unique_pos_tags[pos_after]] = 1

        # Append the vector to the list of vectors
        vectors_pos_tags.append(vector)

    return vectors_pos_tags

#vector that uses the pos tags and lex of the words after and befroe smixut
def make_vector20(smixut_list, unique_lex_words, sentences):
    # Create a list to store the resulting vectors
    combined_vectors = []
    unique_pos_tags = get_unique_pos()

    for i, smixut in enumerate(smixut_list):
        if i >= len(sentences):
            break

        word1, word2 = smixut.split(' ', 1)

        # Initialize the vector with zeros
        vector = [0] * (len(unique_pos_tags) + len(unique_lex_words))

        # Process the word before the smixut
        word_before = find_dict_in_shift_from_smixut(sentences[i], -1, word1)
        if word_before != "err":
            pos_before = word_before.get("pos")
            lex_before = word_before.get("lex")
            if pos_before in unique_pos_tags:
                vector[unique_pos_tags[pos_before]] = 1
            if lex_before in unique_lex_words:
                vector[len(unique_pos_tags) + unique_lex_words[lex_before]] = 1

        # Process the word after the smixut
        word_after = find_dict_in_shift_from_smixut(sentences[i], 1, word2)
        if word_after != "err":
            pos_after = word_after.get("pos")
            lex_after = word_after.get("lex")
            if pos_after in unique_pos_tags:
                vector[unique_pos_tags[pos_after]] = 1
            if lex_after in unique_lex_words:
                vector[len(unique_pos_tags) + unique_lex_words[lex_after]] = 1

        # Append the vector to the list of vectors
        combined_vectors.append(vector)

    return combined_vectors

#vector that uses the pos tags and unqiue of the 2 words before
def make_vector21(smixut_list, unique_words, sentences):
    # Create a list to store the resulting vectors
    combined_vectors = []
    unique_pos_tags = get_unique_pos()

    for i, smixut in enumerate(smixut_list):
        if i >= len(sentences):
            break

        word1, word2 = smixut.split(' ', 1)

        # Initialize the vector with zeros
        vector = [0] * (len(unique_pos_tags) + len(unique_words))

        # Process the word before the smixut
        word_before = find_dict_in_shift_from_smixut(sentences[i], -1, word1)
        if word_before != "err":
            pos_before = word_before.get("pos")
            word_before_text = word_before.get("word")
            if pos_before in unique_pos_tags:
                vector[unique_pos_tags[pos_before]] = 1
            if word_before_text in unique_words:
                vector[len(unique_pos_tags) + unique_words[word_before_text]] = 1

        # Process the word after the smixut
        word_before_2 = find_dict_in_shift_from_smixut(sentences[i], -2, word2)
        if word_before_2  != "err":
            pos_after = word_before_2 .get("pos")
            word_before2_text = word_before_2 .get("word")
            if pos_after in unique_pos_tags:
                vector[unique_pos_tags[pos_after]] = 1
            if word_before2_text in unique_words:
                vector[len(unique_pos_tags) + unique_words[word_before2_text]] = 1

        # Append the vector to the list of vectors
        combined_vectors.append(vector)

    return combined_vectors

#vector that uses the pos tags and lex of the words after and befroe smixut
def make_vector22(smixut_list, unique_lex_words, sentences):
    # Create a list to store the resulting vectors
    combined_vectors = []
    unique_pos_tags = get_unique_pos()

    for i, smixut in enumerate(smixut_list):
        if i >= len(sentences):
            break

        word1, word2 = smixut.split(' ', 1)

        # Initialize the vector with zeros
        vector = [0] * (len(unique_pos_tags) + len(unique_lex_words))

        # Process the word before the smixut
        word_before = find_dict_in_shift_from_smixut(sentences[i], -1, word1)
        if word_before != "err":
            pos_before = word_before.get("pos")
            lex_before = word_before.get("lex")
            if pos_before in unique_pos_tags:
                vector[unique_pos_tags[pos_before]] = 1
            if lex_before in unique_lex_words:
                vector[len(unique_pos_tags) + unique_lex_words[lex_before]] = 1

        # Process the word after the smixut
        word_after = find_dict_in_shift_from_smixut(sentences[i], 1, word2)
        if word_after != "err":
            pos_after = word_after.get("pos")
            lex_after = word_after.get("lex")
            if pos_after in unique_pos_tags:
                vector[unique_pos_tags[pos_after]] = 1
            if lex_after in unique_lex_words:
                vector[len(unique_pos_tags) + unique_lex_words[lex_after]] = 1

        # Append the vector to the list of vectors
        combined_vectors.append(vector)

    return combined_vectors

#vector that uses the pos of the two words before smixut
def make_vector23(smixut_list, sentences):
    # Create a list to store the resulting vectors
    vectors_pos_tags = []

    unique_pos_tags = get_unique_pos()

    for i, smixut in enumerate(smixut_list):
        if i >= len(sentences):
            break


        word1, word2 = smixut.split(' ', 1)
        vector = [0] * len(unique_pos_tags)

        # Find the POS tag of the word before the smixut
        word_before = find_dict_in_shift_from_smixut(sentences[i], -1, word1)
        if word_before != "err":
            pos_before = word_before.get("pos")
            if pos_before in unique_pos_tags:
                vector[unique_pos_tags[pos_before]] = 1

        # Find the POS tag of the word after the smixut
        word_before2 = find_dict_in_shift_from_smixut(sentences[i], -2, word2)
        if word_before2 != "err":
            pos_before2 = word_before2.get("pos")
            if pos_before2 in unique_pos_tags:
                vector[unique_pos_tags[pos_before2]] = 1

        # Append the vector to the list of vectors
        vectors_pos_tags.append(vector)

    return vectors_pos_tags

#vector that uses the pos tags and lex of the two words before smixut
def make_vector24(smixut_list, unique_lex_words, sentences):
    # Create a list to store the resulting vectors
    combined_vectors = []
    unique_pos_tags = get_unique_pos()

    for i, smixut in enumerate(smixut_list):
        if i >= len(sentences):
            break

        word1, word2 = smixut.split(' ', 1)

        # Initialize the vector with zeros
        vector = [0] * (len(unique_pos_tags) + len(unique_lex_words))

        # Process the word before the smixut
        word_before = find_dict_in_shift_from_smixut(sentences[i], -1, word1)
        if word_before != "err":
            pos_before = word_before.get("pos")
            lex_before = word_before.get("lex")
            if pos_before in unique_pos_tags:
                vector[unique_pos_tags[pos_before]] = 1
            if lex_before in unique_lex_words:
                vector[len(unique_pos_tags) + unique_lex_words[lex_before]] = 1

        # Process the word after the smixut
        word_before2 = find_dict_in_shift_from_smixut(sentences[i], -2, word2)
        if word_before2  != "err":
            pos_before2 = word_before2 .get("pos")
            lex_before2 = word_before2 .get("lex")
            if pos_before2 in unique_pos_tags:
                vector[unique_pos_tags[pos_before2]] = 1
            if lex_before2 in unique_lex_words:
                vector[len(unique_pos_tags) + unique_lex_words[lex_before2]] = 1

        # Append the vector to the list of vectors
        combined_vectors.append(vector)
    return combined_vectors

#vector that uses the dep funcs and lex of the two words one before one after smixut
def make_vector25(smixut_list, unique_lex_words, sentences):
    # Create a list to store the resulting vectors
    dep_lex_vectors = []
    unique_dep_func = get_unique_dep_func()

    for i, smixut in enumerate(smixut_list):
        if i >= len(sentences):
            break

        word1, word2 = smixut.split(' ', 1)

        # Initialize the vector with zeros
        vector = [0] * (len(unique_dep_funcs) + len(unique_lex_words))

        # Process the word before the smixut
        word_before = find_dict_in_shift_from_smixut(sentences[i], -1, word1)
        if word_before != "err":
            dep_func_before = word_before.get("dep_func")
            lex_before = word_before.get("lex")
            if dep_func_before in unique_dep_funcs:
                vector[unique_dep_funcs[dep_func_before]] = 1
            if lex_before in unique_lex_words:
                vector[len(unique_dep_funcs) + unique_lex_words[lex_before]] = 1

        # Process the word after the smixut
        word_after = find_dict_in_shift_from_smixut(sentences[i], 1, word2)
        if word_after != "err":
            dep_func_after = word_after.get("dep_func")
            lex_after = word_after.get("lex")
            if dep_func_after in unique_dep_funcs:
                vector[unique_dep_funcs[dep_func_after]] = 1
            if lex_after in unique_lex_words:
                vector[len(unique_dep_funcs) + unique_lex_words[lex_after]] = 1

        # Append the vector to the list of vectors
        dep_lex_vectors.append(vector)

    return dep_lex_vectors

#vector that uses the dep funcs and lex of the two words before smixut
def make_vector26(smixut_list, unique_lex_words, sentences):
    # Create a list to store the resulting vectors
    dep_lex_vectors = []
    unique_dep_funcs = get_unique_dep_func()

    for i, smixut in enumerate(smixut_list):
        if i >= len(sentences):
            break

        word1, word2 = smixut.split(' ', 1)

        # Initialize the vector with zeros
        vector = [0] * (len(unique_dep_funcs) + len(unique_lex_words))

        # Process the word before the smixut
        word_before = find_dict_in_shift_from_smixut(sentences[i], -1, word1)
        if word_before != "err":
            dep_func_before = word_before.get("dep_func")
            lex_before = word_before.get("lex")
            if dep_func_before in unique_dep_funcs:
                vector[unique_dep_funcs[dep_func_before]] = 1
            if lex_before in unique_lex_words:
                vector[len(unique_dep_funcs) + unique_lex_words[lex_before]] = 1

        # Process the word after the smixut
        word_after = find_dict_in_shift_from_smixut(sentences[i], -2, word2)
        if word_after != "err":
            dep_func_after = word_after.get("dep_func")
            lex_after = word_after.get("lex")
            if dep_func_after in unique_dep_funcs:
                vector[unique_dep_funcs[dep_func_after]] = 1
            if lex_after in unique_lex_words:
                vector[len(unique_dep_funcs) + unique_lex_words[lex_after]] = 1

        # Append the vector to the list of vectors
        dep_lex_vectors.append(vector)

    return dep_lex_vectors

#vector that uses the dep funcs of word after and before of smixut
def make_vector27(smixut_list, sentences):
    # Create a list to store the resulting vectors
    dep_func_vectors = []
    unique_dep_funcs = get_unique_dep_func()

    for i, smixut in enumerate(smixut_list):
        if i >= len(sentences):
            break

        word1, word2 = smixut.split(' ', 1)

        # Initialize the vector with zeros
        vector = [0] * (2 * len(unique_dep_funcs))

        # Process the word before the smixut
        word_before = find_dict_in_shift_from_smixut(sentences[i], -1, word1)
        if word_before != "err":
            dep_func_before = word_before.get("dep_func")
            if dep_func_before in unique_dep_funcs:
                vector[unique_dep_funcs[dep_func_before]] = 1

        # Process the word after the smixut
        word_after = find_dict_in_shift_from_smixut(sentences[i], 1, word2)
        if word_after != "err":
            dep_func_after = word_after.get("dep_func")
            if dep_func_after in unique_dep_funcs:
                vector[len(unique_dep_funcs) + unique_dep_funcs[dep_func_after]] = 1

        # Append the vector to the list of vectors
        dep_func_vectors.append(vector)

    return dep_func_vectors

#vector that uses the dep funcs and unique words of the two words one before one after smixut
def make_vector28(smixut_list, unique_words, sentences):
    # Create a list to store the resulting vectors
    dep_func_word_vectors = []
    unique_dep_funcs = get_unique_dep_func()

    for i, smixut in enumerate(smixut_list):
        if i >= len(sentences):
            break

        word1, word2 = smixut.split(' ', 1)

        # Initialize the vector with zeros
        vector = [0] * (2 * len(unique_dep_funcs) + 2 * len(unique_words))

        # Process the word before the smixut
        word_before = find_dict_in_shift_from_smixut(sentences[i], -1, word1)
        if word_before != "err":
            dep_func_before = word_before.get("dep_func")
            word_before_text = word_before.get("word")
            if dep_func_before in unique_dep_funcs:
                vector[unique_dep_funcs[dep_func_before]] = 1
            if word_before_text in unique_words:
                vector[len(unique_dep_funcs) + unique_words[word_before_text]] = 1

        # Process the word after the smixut
        word_after = find_dict_in_shift_from_smixut(sentences[i], 1, word2)
        if word_after != "err":
            dep_func_after = word_after.get("dep_func")
            word_after_text = word_after.get("word")
            if dep_func_after in unique_dep_funcs:
                vector[len(unique_dep_funcs) * 2 + unique_dep_funcs[dep_func_after]] = 1
            if word_after_text in unique_words:
                vector[len(unique_dep_funcs) * 2 + len(unique_words) + unique_words[word_after_text]] = 1

        # Append the vector to the list of vectors
        dep_func_word_vectors.append(vector)

    return dep_func_word_vectors

#vector that uses the dep funcs and unique words of the two words before smixut
def make_vector29(smixut_list, unique_words, sentences):
    # Create a list to store the resulting vectors
    dep_func_word_vectors = []
    unique_dep_funcs = get_unique_dep_func()

    for i, smixut in enumerate(smixut_list):
        if i >= len(sentences):
            break

        word1, word2 = smixut.split(' ', 1)

        # Initialize the vector with zeros
        vector = [0] * (2 * len(unique_dep_funcs) + 2 * len(unique_words))

        # Process the word before the smixut
        word_before = find_dict_in_shift_from_smixut(sentences[i], -1, word1)
        if word_before != "err":
            dep_func_before = word_before.get("dep_func")
            word_before_text = word_before.get("word")
            if dep_func_before in unique_dep_funcs:
                vector[unique_dep_funcs[dep_func_before]] = 1
            if word_before_text in unique_words:
                vector[len(unique_dep_funcs) + unique_words[word_before_text]] = 1

        # Process the word after the smixut
        word_after = find_dict_in_shift_from_smixut(sentences[i], -2, word2)
        if word_after != "err":
            dep_func_after = word_after.get("dep_func")
            word_after_text = word_after.get("word")
            if dep_func_after in unique_dep_funcs:
                vector[len(unique_dep_funcs) * 2 + unique_dep_funcs[dep_func_after]] = 1
            if word_after_text in unique_words:
                vector[len(unique_dep_funcs) * 2 + len(unique_words) + unique_words[word_after_text]] = 1

        # Append the vector to the list of vectors
        dep_func_word_vectors.append(vector)

    return dep_func_word_vectors

#vector that uses the dep funcs of two words before smixut
def make_vector30(smixut_list, sentences):
    # Create a list to store the resulting vectors
    dep_func_vectors = []
    unique_dep_funcs = get_unique_dep_func()

    for i, smixut in enumerate(smixut_list):
        if i >= len(sentences):
            break

        word1, word2 = smixut.split(' ', 1)

        # Initialize the vector with zeros
        vector = [0] * (2 * len(unique_dep_funcs))

        # Process the word before the smixut
        word_before = find_dict_in_shift_from_smixut(sentences[i], -1, word1)
        if word_before != "err":
            dep_func_before = word_before.get("dep_func")
            if dep_func_before in unique_dep_funcs:
                vector[unique_dep_funcs[dep_func_before]] = 1

        # Process the word after the smixut
        word_after = find_dict_in_shift_from_smixut(sentences[i], -2, word2)
        if word_after != "err":
            dep_func_after = word_after.get("dep_func")
            if dep_func_after in unique_dep_funcs:
                vector[len(unique_dep_funcs) + unique_dep_funcs[dep_func_after]] = 1

        # Append the vector to the list of vectors
        dep_func_vectors.append(vector)

    return dep_func_vectors

# def make_parameters():
#     unique_words_lex = {}
#     unique_words = {}
#     unique_genders = { "Masc", "Fem" }
#     unique_pos = {}
#     unique_dep_func = {}
#     smixut_list = []
#     with open('smixut_file.json', 'r', encoding='utf-8') as smixut_file:
#         for smixut in smixut_file:
#             smixut_list.append(smixut)
#         # Read 'output.json' and process the data
#     with open('output.json', 'r', encoding='utf-8') as file:
#         for line in file:
#             if line:
#                 data = json.loads(line)
#                 lst = process_json(data)
#                 sentences.append(lst)
#                 add_unique_words_to_dict(lst, unique_words)
#                 add_unique_words_lex_to_dict(lst, unique_words_lex)
#                 add_unique_pos_to_dict(lst, unique_pos)
#                 add_unique_dep_func_to_dict(lst, unique_dep_func)
#     add_smixut_to_uniqe_words(smixut_list, unique_words)
#
#     return sentences,smixut_list ,unique_words, unique_genders, unique_pos, unique_dep_func, unique_words_lex
#


# sentence = 'בשנת 1948 השלים אפרים קישון את לימודיו בפיסול מתכת ובתולדות האמנות והחל לפרסם מאמרים הומוריסטיים'
# predictions = model.predict([sentence], tokenizer, output_style='json')
# predictions_json = json.dumps(predictions, ensure_ascii=False)
# with open('output.json', 'w', encoding='utf-8') as f:
#     f.write(predictions_json)
#
# # read file
# file_path = 'Sentence_classification.xlsx'
# df = pd.read_excel(file_path)
# column_name = 'משפט'
# column_sentence = []
# column_smixut = []
# sentences = []
# if column_name in df.columns:
#     column_sentence = df[column_name].tolist()
# column_name1 = 'מבנה סמיכות'
# if column_name1 in df.columns:
#     column_smixut = df[column_name1].tolist()
#     unique_words_lex = {}
#     unique_words = {}
#     unique_genders = { "Masc", "Fem" }
#     unique_pos = {}
#     unique_dep_func = {}
#     # with open('output.json', 'w', encoding='utf-8') as f:
#     #     for sentence in column_sentence:
#     #          predictions = model.predict([sentence], tokenizer, output_style='json')
#     #          predictions_json = json.dumps(predictions, ensure_ascii=False)
#     #          f.write(predictions_json+'\n')
#     smixut_list = []
#   #  with open('smixut_file.json', 'w', encoding='utf-8') as smixut_file:
#       #  for smixut in column_smixut:
#          #  smixut_list.append(smixut)
#          #   smixut_json = json.dumps(smixut, ensure_ascii=False)
#         #    smixut_file.write(smixut_json + '\n')
#         # Read 'output.json' and process the data
#     # with open('output.json', 'r', encoding='utf-8') as file:
#     #     for line in file:
#     #         if line:
#     #             data = json.loads(line)
#     #             lst = process_json(data)
#     #             sentences.append(lst)
#     #             add_unique_words_to_dict(lst, unique_words)
#     #             add_unique_words_lex_to_dict(lst, unique_words_lex)
#     #             add_unique_pos_to_dict(lst, unique_pos)
#     #             add_unique_dep_func_to_dict(lst, unique_dep_func)
#
#
#
#     # add_smixut_to_uniqe_words(smixut_list, unique_words)
#     lst = process_json(data)
#     with open('output.txt', 'w', encoding='utf-8') as file1:
#         print_to_file(sentences, file1)

    # with open('unique_words.txt', 'w', encoding='utf-8') as file:
    #     print_to_file(unique_words, file)

    # with open('unique_lex_words.txt', 'w', encoding='utf-8') as file:
    #     print_to_file(unique_words_lex, file)
#     add_smixut_to_uniqe_words(smixut_list, unique_words)
#     lst = process_json(data)
#     with open('output.txt', 'w', encoding='utf-8') as file1:
#         print_to_file(sentences, file1)
#     with open('vec_type_1.txt', 'w', encoding='utf-8') as filetype1:
#         vectors_type1 = make_vector1(smixut_list, unique_words, sentences)
#         for vector in vectors_type1:
#             print_to_file(vector, filetype1)
#     with open('vec_type_2.txt', 'w', encoding='utf-8') as filetype2:
#         vectors_type2 = make_vector2(smixut_list, unique_words_lex, sentences)
#         for vector in vectors_type2:
#             print_to_file(vector, filetype2)

#vectors_type6 = make_vector6(smixut_list, unique_words, sentences)

# vectors_type19 = make_vector19(smixut_list, sentences)
vectors_type28= make_vector7(smixut_list, sentences)


print(vectors_type28)
#print(make_vector7(smixut_list, unique_words, sentences))
# else:
#     print(f"Column '{column_name}' does not exist in the file.")

# def call_to_vector1():
#     sentences, smixut_list, unique_words, unique_genders, unique_pos, unique_dep_func, unique_words_lex = make_parameters()
#     return make_vector1(smixut_list, unique_words, sentences)
#
#
# def call_to_vector2():
#     sentences, smixut_list, unique_words, unique_genders, unique_pos, unique_dep_func, unique_words_lex = make_parameters()
#     return make_vector2(smixut_list, unique_words_lex, sentences)
#
#
# def call_to_vector3():
#     sentences, smixut_list, unique_words, unique_genders, unique_pos, unique_dep_func, unique_words_lex = make_parameters()
#     return make_vector3(unique_words, smixut_list, sentences)
