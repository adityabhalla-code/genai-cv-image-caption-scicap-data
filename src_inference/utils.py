from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pandas as pd
import logging
import ast
import nltk

nltk.download('punkt')
try:
    captions_meta_data = pd.read_excel("data/captions_meta_data_19_may_24.xlsx")
except Exception as e:
    print(f"Exception occured in reading meta data:{e}")

def calculate_bleu(reference, candidate):
    reference = nltk.word_tokenize(reference.lower())
    candidate = nltk.word_tokenize(candidate.lower())
    smoothie = SmoothingFunction().method4
    bleu_score = sentence_bleu([reference], candidate, smoothing_function=smoothie)
    return bleu_score

def get_original_caption(image_id):
    # original_extracted_text = captions_meta_data[captions_meta_data['figure-ID']==fig_id]['0-originally-extracted'].values.tolist()[0]
    return ast.literal_eval(captions_meta_data[captions_meta_data['figure-ID']==image_id]['1-lowercase-and-token-and-remove-figure-index'].values.tolist()[0])['caption']


class Logging:
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(self.log_file_path)
        stream_handler = logging.StreamHandler()
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        logging.basicConfig(level=logging.INFO, handlers=[file_handler, stream_handler])
        self.logger = logging.getLogger(__name__)
            

    
    
    
    
# Example usage
# reference = "The cat is on the mat."
# candidate = "There is a cat on the mat."

# bleu_score = calculate_bleu(reference, candidate)
# print(f"BLEU score: {bleu_score:.4f}")