import requests

url = 'http://192.168.2.69/get?accX&acc_time&accY&accZ'
response = requests.get(url)
data = response.json()

# Extracting the acceleration values
x_acceleration = data['buffer']['accX']['buffer']
y_acceleration = data['buffer']['accY']['buffer']
z_acceleration = data['buffer']['accZ']['buffer']

print(x_acceleration)
