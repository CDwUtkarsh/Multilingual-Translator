import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect
import threading

# Supported models per (src,tgt) language pair
MODEL_MAP = {
    ("en", "hi"): "Helsinki-NLP/opus-mt-en-hi",
    ("hi", "en"): "Helsinki-NLP/opus-mt-hi-en",
    ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
    ("fr", "en"): "Helsinki-NLP/opus-mt-fr-en",
    ("es", "de"): "Helsinki-NLP/opus-mt-es-de",
    ("de", "es"): "Helsinki-NLP/opus-mt-de-es",
    # Add more pairs or use multilingual model for others
}
MULTILINGUAL_MODEL = "facebook/m2m100_418M"

# Cache loaded models
model_cache = {}
lock = threading.Lock()

def load_model(src_lang, tgt_lang):
    key = (src_lang, tgt_lang)
    with lock:
        if key in MODEL_MAP:
            model_name = MODEL_MAP[key]
        else:
            model_name = MULTILINGUAL_MODEL
        if model_name not in model_cache:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            model_cache[model_name] = (tokenizer, model)
        return model_cache[model_name], model_name

def translate(text, src_lang, tgt_lang):
    (tokenizer, model), model_name = load_model(src_lang, tgt_lang)
    if model_name == MULTILINGUAL_MODEL:
        # Special token prep for M2M100
        tokenizer.src_lang = src_lang
        encoded = tokenizer(text, return_tensors="pt")
        generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id(tgt_lang))
        output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    else:
        input_ids = tokenizer.encode(text, return_tensors='pt')
        output_ids = model.generate(input_ids)
        output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output

def detect_language(text):
    return detect(text)
