import time
import openai
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.api_core import retry
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

openai.api_key=os.environ["OPENAI_API"]

model = genai.GenerativeModel(model_name='gemini-1.5-flash')

@retry.Retry()
def request_api_gemini(messages):
    completion = model.generate_content(
        prompt=messages,
            safety_settings=[
            {
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory. HARM_CATEGORY_CIVIC_INTEGRITY: HarmBlockThreshold.BLOCK_NONE,
            }

        ]
    )
    if len(completion.candidates) < 1:
        print(completion)
    ret = completion.candidates[0]['output']
    return ret

def request_api_chatgpt(prompt):
    messages = [
                {"role": "system", "content": 'You are an AI assistant to answer question about biomedicine.'},
                {"role": "user", "content": prompt}
    ]
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages=messages,
        )
        ret = completion["choices"][0]["message"]["content"].strip()
        return ret
    except Exception as E:
        time.sleep(2)
        print(E)
        return request_api_chatgpt(prompt)
