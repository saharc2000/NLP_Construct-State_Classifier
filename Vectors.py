
import json
import pandas as pd
import string

from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('dicta-il/dictabert-tiny-joint')
model = AutoModel.from_pretrained('dicta-il/dictabert-tiny-joint', trust_remote_code=True)

model.eval()

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


def convert_json_to_dict(json_data):
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


def process_json(input_json):
    # Extract tokens from the input JSON
    tokens = input_json[0].get("tokens", [])

    # Convert each token to the desired dictionary format
    result = [convert_json_to_dict(token) for token in tokens]

    return result


def add_unique_words_lex_to_dict(sentence_list, word_dict):
    punctuation = r"\"#$%&'()*+,-–./:;<=>?@[\]^_`{|}~"
    for item in sentence_list:
        lex_word = item.get("lex")
        if not (len(lex_word) == 1 and lex_word in punctuation) and (lex_word and lex_word not in word_dict):
            word_dict[lex_word] = len(word_dict)
    return word_dict

def add_unique_words_to_dict(sentence_list, word_dict):
    punctuation = r"\"#$%&'()*+,-–./:;<=>?@[\]^_`{|}~"
    for item in sentence_list:
        lex_word = item.get("word")
        if not (len(lex_word) == 1 and lex_word in punctuation) and (lex_word and lex_word not in word_dict):
            word_dict[lex_word] = len(word_dict)
    return word_dict

def get_lex(word):
    return word # complete later

def make_vector1(vectors, smixut_list, unique_words,sentences):
    #create list of vectors
    for smixut in smixut_list:
        word1, word2 = smixut.split(' ')
        # find_token_info()
        vector = [0] * len(unique_words)
        # complete get_lex
        vector[unique_words[get_lex(word1)]] = 1
        vector[unique_words[get_lex(word2)]] = 1
        vectors.append(vector)



# sentence = 'בשנת 1948 השלים אפרים קישון את לימודיו בפיסול מתכת ובתולדות האמנות והחל לפרסם מאמרים הומוריסטיים'
# predictions = model.predict([sentence], tokenizer, output_style='json')
# predictions_json = json.dumps(predictions, ensure_ascii=False)
# with open('output.json', 'w', encoding='utf-8') as f:
#     f.write(predictions_json)

# read file
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
    # with open('output.json', 'w', encoding='utf-8') as f:
    #     for sentence in column_sentence:
    #         predictions = model.predict([sentence], tokenizer, output_style='json')
    #         predictions_json = json.dumps(predictions, ensure_ascii=False)
    #         f.write(predictions_json+'\n')
    smixut_list = []
    with open('smixut_file.json', 'w', encoding='utf-8') as smixut_file:
        for smixut in column_smixut:
            smixut_list.append(smixut)
            smixut_json = json.dumps(smixut, ensure_ascii=False)
            smixut_file.write(smixut_json + '\n')

        # Read 'output.json' and process the data
    with open('output.json', 'r', encoding='utf-8') as file:
        for line in file:
            if line:
                data = json.loads(line)
                lst = process_json(data)
                sentences.append(lst)
                add_unique_words_to_dict(lst, unique_words)
                add_unique_words_lex_to_dict(lst, unique_words_lex)

    with open('unique_words.txt', 'w', encoding='utf-8') as file:
        print_to_file(unique_words, file)


    lst = process_json(data)
    with open('output.txt', 'w', encoding='utf-8') as file1:
        print_to_file(lst, file1)



# else:
#     print(f"Column '{column_name}' does not exist in the file.")
