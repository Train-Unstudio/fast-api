import requests
from io import BytesIO
from PIL import Image

# URL of the API endpoint
url = "http://localhost:8000/inpaint"

# Read the input image
image = Image.open("fastapi/images.jpg")

# Encode the image as bytes and create a dictionary with the request parameters
image_bytes = BytesIO()
image.save(image_bytes, format="PNG")
image_bytes.seek(0)
files = {"image_file": ("input.png", image_bytes, "image/png")}
data = {"prompt": "Inpaint the image", "num_steps": 10}

# Send the POST request
response = requests.post(url, files=files, data=data)

# Check if the request was successful
if response.status_code == 200:
    # Decode the output image
    output_image_bytes = BytesIO(response.content)
    output_image = Image.open(output_image_bytes)
    # Display the output image
    output_image.show()
else:
    print("Error:", response.status_code, response.text)
