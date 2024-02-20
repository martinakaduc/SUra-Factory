import sys
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

def compare_models(model_1, model_2):
    models_differ = 0
    total = 0
    identical = []
    
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        total += 1
        if torch.equal(key_item_1[1], key_item_2[1]):
            identical.append(key_item_1[0])
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')

    print("Number of different modules:", models_differ)
    print("Total:", total)
    print(identical)
          
model_1 = AutoModelForCausalLM.from_pretrained(sys.argv[1])
model_2 = AutoModelForCausalLM.from_pretrained(sys.argv[2])

compare_models(model_1, model_2)