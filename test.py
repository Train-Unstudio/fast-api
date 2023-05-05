import requests
import base64
url = 'http://localhost:8000/design/sd'

def img2base64():
    with open("/content/serum_standard.png","rb") as img_file:
      encoded_string=base64.b64encode(img_file.read()).decode("utf-8")
    return encoded_string
b64_string=img2base64()

data = {
    'base64_string': b64_string,
    'prompt': '...your prompt here...'
}
response = requests.post(url, data=data)
print(response.json())

