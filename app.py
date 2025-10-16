import os
import streamlit as st
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import threading
import utils
import uvicorn

# Translation endpoint schema
class TranslateRequest(BaseModel):
    text: str
    src_lang: str
    tgt_lang: str

app = FastAPI()
translation_history = []

@app.post('/translate')
async def translate_endpoint(request: TranslateRequest):
    try:
        result = utils.translate(request.text, request.src_lang, request.tgt_lang)
        translation_history.append({
            "input": request.text,
            "output": result,
            "from": request.src_lang,
            "to": request.tgt_lang
        })
        return {"translated_text": result}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post('/detect')
async def detect_endpoint(request: Request):
    data = await request.json()
    lang = utils.detect_language(data["text"])
    return {"lang": lang}

def run_backend():
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Thread to keep backend running beside Streamlit
threading.Thread(target=run_backend, daemon=True).start()

# Streamlit Frontend
st.set_page_config(page_title="Multilingual Machine Translator", layout="centered")

st.title("🌐 Multilingual Machine Translator")
st.write("Translate text between multiple languages using Transformer models!")

LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
}

def get_language_label(code):
    for name, c in LANGUAGES.items():
        if c == code:
            return name
    return code

if "history" not in st.session_state:
    st.session_state["history"] = []

# UI Elements
input_text = st.text_area("Enter text", "", height=100)
col1, col2 = st.columns(2)
with col1:
    src_lang = st.selectbox("Source Language", list(LANGUAGES.keys()), index=0)
with col2:
    tgt_lang = st.selectbox("Target Language", list(LANGUAGES.keys()), index=1)
src_code = LANGUAGES[src_lang]
tgt_code = LANGUAGES[tgt_lang]

swap = st.button("Swap Languages")
if swap:
    src_lang, tgt_lang = tgt_lang, src_lang
    src_code, tgt_code = tgt_code, src_code

auto_detect = st.checkbox("Auto-detect source language")
dark_mode = st.checkbox("Dark Mode")

if dark_mode:
    st.markdown("<style>body{background-color:#1a1a1a;color:white;}</style>", unsafe_allow_html=True)

translate_btn = st.button("Translate", disabled=not input_text.strip())
clear_btn = st.button("Clear")
translated = ""

if clear_btn:
    input_text = ""
    st.session_state["translated_text"] = ""

if auto_detect and input_text.strip():
    import requests
    res = requests.post("http://localhost:8000/detect", json={"text": input_text})
    detected = res.json().get("lang", "")
    if detected in LANGUAGES.values():
        src_code = detected
        label = get_language_label(detected)
        st.info(f"Detected language: {label}")

if translate_btn and input_text.strip():
    with st.spinner("Translating..."):
        import requests
        res = requests.post("http://localhost:8000/translate", json={
            "text": input_text,
            "src_lang": src_code,
            "tgt_lang": tgt_code,
        })
        if "translated_text" in res.json():
            translated = res.json()["translated_text"]
            st.session_state["translated_text"] = translated
            st.session_state["history"].append({
                "input": input_text,
                "from": src_code,
                "to": tgt_code,
                "output": translated
            })
        else:
            st.error("Translation error: " + res.json().get("error", "Unknown error"))

if "translated_text" in st.session_state:
    st.text_area("Translation", st.session_state["translated_text"], height=100)

    # Optional TTS
    speak_btn = st.button("Speak")
    if speak_btn:
        from gtts import gTTS
        tts = gTTS(text=st.session_state["translated_text"], lang=tgt_code)
        tts.save("tts_output.mp3")
        audio_file = open("tts_output.mp3", "rb")
        st.audio(audio_file.read(), format="audio/mp3")

    copy_btn = st.button("Copy Translation")
    if copy_btn:
        st.write("Copied to clipboard! (Note: In browser, use Ctrl+C)")

st.header("Translation History")
for item in st.session_state["history"][-5:][::-1]:
    st.write(f'{get_language_label(item["from"])} ➔ {get_language_label(item["to"])}')
    st.write(f'Input: {item["input"]}')
    st.write(f'Output: {item["output"]}')
    st.write("---")
