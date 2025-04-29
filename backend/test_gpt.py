# backend/test_gpt.py

import os
import openai
from dotenv import load_dotenv

# Load your OPENAI_API_KEY from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_summary(sentences):
    """
    Given a list of detected words (e.g. ["hello", "iloveyou"]),
    ask ChatGPT to produce three example English sentences
    that each include all those words, adding any filler words
    needed to make them grammatically correct.
    Returns them as a numbered list string.
    """
    if not sentences:
        return ""

    # Build prompt with your detected words
    prompt = (
        "You are an AI assistant.  "
        "I have detected the following English words from American Sign Language gestures: "
        f"{', '.join(sentences)}.  "
        "Please generate three, short, natural-sounding English sentences.  "
        "Each sentence must include all of the detected words, be short and concise, "
        "and only add words needed to make the sentences flow.  "
        "Return your answer as a numbered list, one sentence per item."
    )

    try:
        resp = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an ASL translation assistant."},
                {"role": "user",   "content": prompt}
            ],
            temperature=0.7,
            max_tokens=120
        )
        # The raw content will be something like:
        # 1. Hello, and I love you!
        # 2. I love you, and I say Hello!
        # 3. Hello, I love you!
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ ChatGPT API error in generate_summary: {e}")
        return ""
