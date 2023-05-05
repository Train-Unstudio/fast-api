import requests

url = 'http://localhost:8000/design/sd'
data = {
    'base64_string': '...your base64 string here...',
    'prompt': '...your prompt here...'
}
response = requests.post(url, data=data)
print(response.json())
