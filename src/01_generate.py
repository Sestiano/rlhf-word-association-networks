# 01_generate.py - Word Association Generation
import json, requests, re, sys, os, time
from spacy.lang.en.stop_words import STOP_WORDS as STOPWORDS
from config import *

def generate_base(cue, retries=5, seed=None):
    """Completion API for base model (few-shot)"""
    prompt = PROMPT_BASE.format(cue=cue)
    for attempt in range(retries):
        try:
            payload = {
                "model": MODEL_BASE, "prompt": prompt,
                "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE,
                "stop": STOP
            }
            if seed is not None:
                payload["seed"] = seed
            r = requests.post(LM_STUDIO_BASE, json=payload, timeout=120)
            data = r.json()
            if "choices" in data and data["choices"][0]["text"].strip():
                return data["choices"][0]["text"].strip()
            print(f"  empty response, retry {attempt+1}/{retries}")
        except Exception as e:
            print(f"  error: {e}, retry {attempt+1}/{retries}")
        time.sleep(5)
    raise RuntimeError(f"Base API failed after {retries} retries")

def generate_chat(cue, retries=5, seed=None):
    """Chat API for chat model (system + user)"""
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": USER_MSG.format(cue=cue)}
    ]
    for attempt in range(retries):
        try:
            payload = {
                "model": MODEL_CHAT, "messages": messages,
                "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE
            }
            if seed is not None:
                payload["seed"] = seed
            r = requests.post(LM_STUDIO_CHAT, json=payload, timeout=120)
            data = r.json()
            if "choices" in data and data["choices"][0]["message"]["content"].strip():
                return data["choices"][0]["message"]["content"].strip()
            print(f"  empty response, retry {attempt+1}/{retries}")
        except Exception as e:
            print(f"  error: {e}, retry {attempt+1}/{retries}")
        time.sleep(5)
    raise RuntimeError(f"Chat API failed after {retries} retries")

def parse_words(text):
    """Extract words, filter stopwords/duplicates/cue"""
    # handles 'word, word', 'word\nword', and '1. word' formats
    text = re.sub(r'\d+\.\s*', ', ', text)  # remove numbering
    words = re.split(r'[,\n]+', text)
    seen = set()
    result = []
    for w in words:
        w = w.strip().lower()
        # keep only single alphabetic words
        if w and w.isalpha() and len(w) >= 2 and w not in STOPWORDS and w not in seen:
            seen.add(w)
            result.append(w)
        if len(result) >= 10:
            break
    return result

if __name__ == "__main__":
    # --test: 1 sample, 2 cues per valence (6 calls, ~15 sec)
    test_mode = '--test' in sys.argv
    n_samples = 1 if test_mode else N_SAMPLES
    if test_mode:
        print("TEST MODE: 1 sample, 2 cue per valenza")
        cues = {v: ws[:2] for v, ws in CUES.items()}
    else:
        cues = CUES

    # Resume: load partial results
    import os
    path = "data/associations.json"
    if os.path.exists(path):
        with open(path) as f:
            results = json.load(f)
        print("Resuming from partial results")
    else:
        results = {"base": {}, "chat": {}}

    call_idx = 0  # global counter for incremental seed

    for valence, words in cues.items():
        for cue in words:
            key = f"{valence}_{cue}"
            if key in results["base"] and len(results["base"][key]) >= n_samples:
                print(f"{valence}/{cue}... skip")
                continue

            if key not in results["base"]:
                results["base"][key] = []
                results["chat"][key] = []

            done = len(results["base"][key])

            print(f"{valence}/{cue}... ({done}/{n_samples})")
            for _ in range(done, n_samples):
                # Base model (completion API)
                text = generate_base(cue, seed=SEED + call_idx)
                call_idx += 1
                results["base"][key].append({"cue": cue, "valence": valence, "words": parse_words(text)})

                # Chat model (chat API)
                text = generate_chat(cue, seed=SEED + call_idx)
                call_idx += 1
                results["chat"][key].append({"cue": cue, "valence": valence, "words": parse_words(text)})

            with open(path, "w") as f:
                json.dump(results, f, indent=2)

    print(f"Saved {path}")
