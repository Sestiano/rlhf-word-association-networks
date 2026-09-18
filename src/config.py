# config.py - Word Association Task Configuration
LM_STUDIO_BASE = "http://localhost:1234/v1/completions"
LM_STUDIO_CHAT = "http://localhost:1234/v1/chat/completions"
MODEL_BASE = "thebloke/llama-2-7b-gguf"
MODEL_CHAT = "thebloke/llama-2-7b-chat-gguf"
MAX_TOKENS = 60
TEMPERATURE = 0.7
STOP = ["\n\n", "\n1", "\nList"]
SEED = 42
N_SAMPLES = 50

# Few-shot prompt for base model (completion)
PROMPT_BASE = """List 10 words associated with 'dog': bone, bark, tail, leash, puppy, fetch, collar, paw, fur, walk
List 10 words associated with 'ocean': wave, salt, fish, tide, shore, deep, coral, whale, sand, swim
List 10 words associated with '{cue}':"""

# System + user messages for chat model
SYSTEM_MSG = "You are a word association assistant. When given a word, respond ONLY with exactly 10 associated words separated by commas. No explanations, no numbering."
USER_MSG = "List 10 words associated with '{cue}':"

# Cue words: 10 positive, 10 negative, 10 neutral
# Concrete/semi-concrete, no direct Plutchik emotions
# Valence verified with EmoAtlas z-scores
# Ref: ANEW (Bradley & Lang 1999), Brysbaert et al. 2014
CUES = {
    "positive": ["treasure", "medal", "bloom", "embrace", "gift",
                 "birthday", "vacation", "harvest", "sunshine", "rainbow"],
    "negative": ["poison", "disaster", "prison", "plague", "wound",
                 "betrayal", "exile", "darkness", "corpse", "ruin"],
    "neutral": ["ladder", "mirror", "wheel", "envelope", "bridge",
                "thread", "anchor", "lantern", "bucket", "rope"]
}

# Completion-style prompt (used only as fallback)
PROMPT_TEMPLATE = USER_MSG

NEGATIVE_EMOTIONS = {'fear', 'anger', 'sadness', 'disgust'}
