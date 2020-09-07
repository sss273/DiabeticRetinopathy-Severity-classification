import base64
from urllib.parse import urlencode
from urllib.request import Request, urlopen

with open("./test_data/0a0780ad3395.png", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read())

print(encoded_string)

url = 'http://127.0.0.1:5000/detect' # Set destination URL here
post_fields = {'image': encoded_string}     # Set POST fields here

request = Request(url, urlencode(post_fields).encode())
json = urlopen(request).read().decode()
print(json)
