from openai import OpenAI

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="bananasplitsapikey",
)



import base64

def encode_image_to_data_uri(image_path):
    with open(image_path, "rb") as image_file:
        encoded_str = base64.b64encode(image_file.read()).decode("utf-8")
    # Adjust the MIME type if your image is not PNG
    return f"data:image/png;base64,{encoded_str}"

image_data_uri = encode_image_to_data_uri(r"C:\Users\Richard\OneDrive\GIT\CoMix\data\datasets.unify\2000ad\images\1aba4642\199.jpg")

completion = client.chat.completions.create(
    extra_headers={
        "HTTP-Referer": "",
        "X-Title": "FASERIPing",
    },
    extra_body={},
    #model="google/gemma-3-27b-it:free",
    #model="qwen/qwen2.5-vl-32b-instruct:free",
    model="qwen/qwen2.5-vl-32b-instruct:free",
    #qwen/qwen2.5-vl-72b-instruct:free
    temperature=1.0,   # float between 0 and 1+ controlling randomness; lower = more deterministic
    top_p=1,    
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Please read, split out the panels, caption them, identify the speakers."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data_uri
                    }
                }
            ],
        }
    ],
)

print(completion.choices[0].message.content)
