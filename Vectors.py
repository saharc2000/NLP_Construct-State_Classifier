
import json
import pandas as pd
import torch

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

sentence = 'בשנת 1948 השלים אפרים קישון את לימודיו בפיסול מתכת ובתולדות האמנות והחל לפרסם מאמרים הומוריסטיים'
predictions = model.predict([sentence], tokenizer, output_style='json')
predictions_json = json.dumps(predictions, ensure_ascii=False)

with open('output.json', 'w', encoding='utf-8') as f:
    f.write(predictions_json)

# read file
file_path = 'Sentence_classification.xlsx'
df = pd.read_excel(file_path)

column_name = 'משפט'
if column_name in df.columns:
    column_content = df[column_name].tolist()
    with open('output.json', 'w', encoding='utf-8') as f:
        sentence = "כאבי בטן"
        # for sentence in column_content:
        predictions = model.predict([sentence], tokenizer, output_style='json')
        predictions_json = json.dumps(predictions, ensure_ascii=False)
        f.write(predictions_json+'\n')
    
    with open('output.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    word = "כאבי בטן"
    word1, word2 = word.split(' ')
    
    tokens1 = find_token_info(data, word1)
    tokens2 = find_token_info(data, word2)
    with open('output.txt', 'w', encoding='utf-8') as file:
        print_to_file(tokens1, file)    
        print_to_file(tokens2, file)    

else:
    print(f"Column '{column_name}' does not exist in the file.")

