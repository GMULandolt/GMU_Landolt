import sys
import json
import base64
import requests

# Define API URL and SPK filename:
url = 'https://ssd.jpl.nasa.gov/api/horizons_file.api'
spk_filename = 'spk_file.bsp'

# Get the Horizons API input file from the command-line:
f = open("input.txt")

# Build and submit the API request and decode the JSON-response:
response = requests.post(url, data={'format':'json'}, files={'input': f})
f.close()
data = json.loads(response.text)

spk_filename = data["spk_file_id"] + ".bsp"
f = open(spk_filename, "wb")
#
# Decode and write the binary SPK file content:
final = base64.b64decode(data["spk"])
f.write(final)
f.close()
print("wrote SPK content to {0}".format(spk_filename))
sys.exit()