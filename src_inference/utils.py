from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pandas as pd
from app_logger import log
import ast
import nltk



nltk.download('punkt')
try:
    captions_meta_data = pd.read_excel("data/captions_meta_data_19_may_24.xlsx")
except Exception as e:
    log.error(f"Exception occured in reading meta data:{e}")
    log.error(str(e))

def calculate_bleu(reference, candidate):
    if reference == "none":
        return "can't compute belus as reference is none!"
    reference = nltk.word_tokenize(reference.lower())
    candidate = nltk.word_tokenize(candidate.lower())
    smoothie = SmoothingFunction().method4
    bleu_score = sentence_bleu([reference], candidate, smoothing_function=smoothie)
    return bleu_score

def get_original_caption(image_id):
    # original_extracted_text = captions_meta_data[captions_meta_data['figure-ID']==fig_id]['0-originally-extracted'].values.tolist()[0]
    try:
        return ast.literal_eval(captions_meta_data[captions_meta_data['figure-ID']==image_id]['1-lowercase-and-token-and-remove-figure-index'].values.tolist()[0])['caption']
    except Exception as e:
        log.error(f"Exception occured in getting original caption:{e}")
        log.error(str(e))
        return "none"






# Example usage
# reference = "The cat is on the mat."
# candidate = "There is a cat on the mat."
# bleu_score = calculate_bleu(reference, candidate)
# print(f"BLEU score: {bleu_score:.4f}")