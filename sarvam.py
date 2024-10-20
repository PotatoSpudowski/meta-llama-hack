import requests

API = "b77626ab-27db-46e8-852d-297aba19e24b"

url = "https://api.sarvam.ai/translate"

language_codes = [
    "hi-IN",
    "bn-IN",
    "kn-IN",
    "ml-IN",
    "mr-IN",
    "od-IN",
    "pa-IN",
    "ta-IN",
    "te-IN",
    "gu-IN",
]


payload_template = {
    "input": "How are you doing today?",
    "source_language_code": "en-IN",
    "speaker_gender": "Male",
    "mode": "formal",
    "model": "mayura:v1",
    "enable_preprocessing": True,
}

headers = {"Content-Type": "application/json", "api-subscription-key": API}

# Loop through each language code
for lang_code in language_codes:
    # Create a copy of the payload and update the target language code
    payload = payload_template.copy()
    payload["target_language_code"] = lang_code

    print(payload)
    # Make the POST request
    response = requests.post(url, json=payload, headers=headers)

    # Print the response for each language
    print(f"Translation for {lang_code}: {response.text}")
