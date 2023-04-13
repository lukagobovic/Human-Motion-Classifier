import requests

url = 'http://10.216.180.159/get?accX=full&acc_time&accY&accZ'
response = requests.get(url)
data = response.json()

while True:
  # Extracting the acceleration values
  x_acceleration = data['buffer']['accX']['buffer']
  y_acceleration = data['buffer']['accY']['buffer']
  z_acceleration = data['buffer']['accZ']['buffer']

  print(x_acceleration)
