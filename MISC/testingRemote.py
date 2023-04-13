import numpy as np
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.common.by import By
import time
from scipy.stats import skew
import joblib
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
import pickle
import requests

#global var
driver = None

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

window_size = 5
# Define the preprocessing function
def preprocess_data(data):
  data = data.rolling(window_size).mean().dropna()
  # Remove outliers
  data = data[(np.abs(data - data.mean()) / data.std()) < 3]

  # Normalize the data
  data = (data - data.mean()) / data.std()

  # Extract features
  features = [
    np.max(data),
    np.min(data),
    np.ptp(data),
    np.mean(data),
    np.median(data),
    np.var(data),
    np.std(data)
  ]

  return features


url = 'http://192.168.2.69/get?accX&acc_time&accY&accZ'

window_size = 50
iterations = 15

def process_data_and_predict():
    for n in range(iterations):
        # Create an empty array with shape (1, 100, 4)
        x_data_array = np.zeros(window_size)
        y_data_array = np.zeros(window_size)
        z_data_array = np.zeros(window_size)
        total_data_array = np.zeros(window_size)

        # Initialize index to keep track of data points
        index = 0

        while True:
            response = requests.get(url)
            data = response.json()

            x_acceleration = data['buffer']['accX']['buffer']
            y_acceleration = data['buffer']['accY']['buffer']
            z_acceleration = data['buffer']['accZ']['buffer']
            value_total = np.sqrt(x_acceleration[0] ** 2 + y_acceleration[0] ** 2 + z_acceleration[0] ** 2)

            # Convert values to floats and store in data_array
            x_data_array.append(x_acceleration)
            y_data_array.append(y_acceleration)
            z_data_array.append(z_acceleration)
            total_data_array[index] = value_total
            index += 1
            # Break loop after 30 data points
            if index == window_size:
                break


        x_data_features = preprocess_data(x_data_array)
        y_data_features = preprocess_data(y_data_array)
        z_data_features = preprocess_data(z_data_array)
        total_data_features = preprocess_data(total_data_array)

        X_combined = np.concatenate((x_data_features, y_data_features, z_data_features, total_data_features), axis=0)
        X_combined = X_combined.reshape(1, -1)

        Y_predicted = model.predict(X_combined)
        print(Y_predicted)
        change_window_color(Y_predicted[0])
        

#functions for UI
def get_webdriver_path():
    global driver
    webdriver_path = filedialog.askopenfilename()
    driver = webdriver.Chrome(webdriver_path)


def get_ip_address():
    global driver
    ip_address = ip_address_entry.get()
    if ip_address:
        driver = webdriver.Chrome()
        driver.get(f'http://{ip_address}/')
    else:
        messagebox.showwarning("Warning", "Please enter a valid IP address.")


def show_instructions():
    instructions = """1. Open the Phyphox app
2. Click on "Acceleration without G"
3. Activate the "Access with distance" option by clicking the three buttons in the top right corner of the Phyphox interface
4. Input the URL provided by Phyphox into the UI text box and go to it on Chrome
5. Input the location of the web driver into the UI text box"""

    messagebox.showinfo("Instructions", instructions)


def change_window_color(predicted_output):
    if large_window:
        if predicted_output == 'walking':
            large_window.configure(bg='red')
        else:
            large_window.configure(bg='blue')

#UI
window = tk.Tk()
window.configure(bg='black')
window.title('Acceleration Classifier')

webdriver_path_var = tk.StringVar()
webdriver_path_entry = tk.Entry(window, textvariable=webdriver_path_var, width=50)
webdriver_path_entry.pack()
webdriver_path_entry.place(x=100, y=30)

webdriver_path_button = tk.Button(window, text="Select Webdriver", command=get_webdriver_path, bg='white', fg='black', font=('Arial', 10, 'bold'), width=20)
webdriver_path_button.pack()
webdriver_path_button.place(x=430, y=30)

ip_address_entry = tk.Entry(window, width=50)
ip_address_entry.pack()
ip_address_entry.place(x=100, y=70)

ip_address_button = tk.Button(window, text="Enter IP Address", command=lambda: [process_data_and_predict()], bg='white', fg='black', font=('Arial', 10, 'bold'), width=20)
ip_address_button.pack()
ip_address_button.place(x=430, y=70)

instructions_button = tk.Button(window, text="Instructions", command=show_instructions, bg='white', fg='black', font=('Arial', 10, 'bold'), width=20)
instructions_button.pack()
instructions_button.place(x=430, y=110)

#legend for large_window
legend_label = tk.Label(window, text="Legend: Red = Walking, Blue = Jumping", bg='black', fg='white', font=('Arial', 12, 'bold'))
legend_label.pack()
legend_label.place(x=300, y=150)

large_window = tk.Frame(window, bg='white', width=800, height=350)
large_window.pack()
large_window.place(x=100, y=180)

window.geometry("1000x600")
window.mainloop()