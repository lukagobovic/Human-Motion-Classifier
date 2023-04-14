import os
from tkinter import *
from tkinter import filedialog

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
NavigationToolbar2Tk)
from PIL import ImageTk, Image
import numpy as np
import pickle
from scipy.stats import skew,kurtosis
from sklearn.preprocessing import MinMaxScaler

plot_canvas = None

def browse_file():
    filename = filedialog.askopenfilename()
    csv_input.delete(first=0, last=255)
    csv_input.insert(0, filename)

def import_csv():
    filename = csv_input.get()
    if filename == "":
        csv_status.config(text="Invalid Input")
        return
    predict_button.config(state="disabled")
    browse_button.config(state="disabled")
    csv_input.config(state="disabled")
    csv_status.config(text="Processing...")

    try:
        csv_file = pd.read_csv(filename)
        csv_file = csv_file.iloc[:,1:]
    except:
        csv_status.config(text="File not found")
        predict_button.config(state="normal")
        csv_input.config(state="normal")
        return
    else:
        csv_status.config(text="File found, Reading file...")
        csv_label.grid(row=2, column=0, padx=10)
        prediciton = predict(csv_file,filename)
        plot(plot_canvas,csv_file,prediciton)
        # plot_canvas.destroy()

def preprocess_data(data):
  window_size = 50
  num_windows = len(data) // window_size
  features = []
  for i in range(num_windows):
    window_data = data.iloc[i*window_size:(i+1)*window_size]
    window_features = [
        np.max(window_data.x),
        np.min(window_data.x),
        np.ptp(window_data.x),
        np.mean(window_data.x),
        np.median(window_data.x),
        skew(window_data.x),
        np.var(window_data.x),
        np.std(window_data.x),
        kurtosis(window_data.x),
        np.sqrt(np.mean(window_data.x ** 2)),
        np.max(window_data.y),
        np.min(window_data.y),
        np.ptp(window_data.y),
        np.mean(window_data.y),
        np.median(window_data.y),
        skew(window_data.y),
        np.var(window_data.y),
        np.std(window_data.y),
        kurtosis(window_data.y),
        np.sqrt(np.mean(window_data.y ** 2)),
        np.max(window_data.z),
        np.min(window_data.z),
        np.ptp(window_data.z),
        np.mean(window_data.z),
        np.median(window_data.z),
        skew(window_data.z),
        np.var(window_data.z),
        np.std(window_data.z),
        kurtosis(window_data.z),
        np.sqrt(np.mean(window_data.z ** 2)),
        np.max(window_data.total_acceleration),
        np.min(window_data.total_acceleration),
        np.ptp(window_data.total_acceleration),
        np.mean(window_data.total_acceleration),
        np.median(window_data.total_acceleration),
        skew(window_data.total_acceleration),
        np.var(window_data.total_acceleration),
        np.std(window_data.total_acceleration),
        kurtosis(window_data.total_acceleration),
        np.sqrt(np.mean(window_data.total_acceleration ** 2))
    ]
    features.append(window_features)

  return features

# Function to predict whether inputted data is walking or jumping
def predict(csvFile,filename):

    with open('../model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Load the original data
    originalData = csvFile
    window_size = 5

    df = originalData
    df = df.iloc[:-(df.shape[0] % 500)+(window_size-1)] 
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR)))]

    #Interpolate the missing values
    df.interpolate(method='linear', inplace=True)
    
    #Rolling average filter
    df = df.rolling(5).mean().dropna()
        
    # Normalize the data
    scaler = MinMaxScaler()
    df = scaler.fit_transform(df)
    df = pd.DataFrame(df)
    data = df

    # Segment the data into windows of a given size
    def segment_data(data, window_size):
        num_samples = int(np.floor(data.shape[0] / (window_size * 100)))  # 100 Hz sampling rate
        segments = []
        for i in range(num_samples):
            start_idx = i * window_size * 100
            end_idx = start_idx + window_size * 100
            segment = data.iloc[start_idx:end_idx]
            segmentNP = segment.to_numpy()
            np.random.shuffle(segmentNP)
            segment = pd.DataFrame(segmentNP)
            segments.append(segment)
        return segments

    # Segment the data and shuffle it
    segments = segment_data(data, window_size)
    np.random.shuffle(segments)
    for df in segments:
        df.rename(columns={0: "x", 1: "y", 2: "z", 3: "total_acceleration"}, inplace=True)

    # Extract features
    train_features = np.concatenate([preprocess_data(segment) for segment in segments])

    # Make predictions
    y_pred = model.predict(train_features)
    plt.plot(y_pred)
    plt.show()

    predicted_class = np.bincount(y_pred.astype(int)).argmax()
    if predicted_class == 0:
        print("This CSV contains walking data.")
        originalData['Walking(0)/Jumping(1)'] = 0
        originalData.to_csv('../OutputData/'+os.path.basename(filename)+'output'+'.csv', index=False)
    else:
        print("This CSV contains jumping data.")
        originalData['Walking(0)/Jumping(1)'] = 1
        originalData.to_csv('../OutputData/'+os.path.basename(filename)+'output'+'.csv', index=False)

    return predicted_class

def plot(canvas,data,prediciton):
    # the figure that will contain the plot
    fig = Figure(figsize=(5, 5),dpi=100)
    # adding the subplot
    plot1 = fig.add_subplot(111)
    # plotting the graph
    plot1.plot(data.iloc[1000:1500,:])
    # creating the Tkinter canvas
    # containing the Matplotlib figure
    canvas = FigureCanvasTkAgg(fig, master=program_frame)  # A tk.DrawingArea.
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=1, rowspan=10, padx=25, pady=25)
    if prediciton == 0:    
        result_label.config(text="RESULT: Walking Motion")
    elif prediciton == 1:
        result_label.config(text="RESULT: Jumping Motion")

def reset():
    global plot_canvas  # Access the global variable plot_canvas

    predict_button.config(state="normal")
    browse_button.config(state="normal")
    csv_input.config(state="normal")
    csv_input.delete(first=0, last=255)
    result_label.config(text="")
    csv_status.config(text="")
    
    # Create a new blank plot canvas and display it
    fig = Figure(figsize=(5, 5), dpi=100)
    plot_canvas = FigureCanvasTkAgg(fig, master=program_frame)
    plot_canvas.get_tk_widget().grid(row=0, column=1, rowspan=10, padx=25, pady=25)
    plot_canvas.grid(row=0, column=1, rowspan=10, padx=25, pady=25)
    plot_canvas.draw()

# Graphical User Interface ------------------------------------------------------------

window = Tk()
window.title("Human Motion Classifier")
window.geometry("1080x720")
window.config(bg="#212121")


# Import Images using Pillow
icon_file = ImageTk.PhotoImage(Image.open("sci24crest.png"))
window.wm_iconphoto(False, icon_file)
browse_img = ImageTk.PhotoImage(Image.open("button_browse.png"))
predict_img = ImageTk.PhotoImage(Image.open("button_predict.png"))
reset_img = ImageTk.PhotoImage(Image.open("button_reset.png"))
exit_img = ImageTk.PhotoImage(Image.open("button_exit-program.png"))

program_frame = Frame(window, highlightbackground="#9900ff", highlightcolor="#9900ff",
                      background="#424242", highlightthickness=4, width=1080, height=720)
input_frame = Frame(program_frame, background="#424242")

# Define Frames & Sub frames
title = Label(text="Human Motion Classifier", font=("Open Sans Bold",36), fg="#9900ff", bg="#212121")
credits = Label(text="Created by Group 53 - Bennett Desmarais, CJ Akkawi, Luka Gobovic", font=("Open Sans", 12), fg="white", bg="#212121")
csv_label = Label(program_frame, text="Enter the file directory of the CSV file:", font=("Open Sans", 14), fg="white", bg="#424242")
csv_input = Entry(input_frame, width=40, font=("Open Sans", 12))
csv_status = Label(program_frame, text="", font=("Open Sans", 14), fg="white", bg="#424242")
result_label = Label(program_frame, text="", font=("Open Sans", 14), fg="white", bg="#424242")
browse_button = Button(input_frame, image=browse_img, command=browse_file, borderwidth=0, bg="#424242", activebackground="#424242")
predict_button = Button(program_frame, image=predict_img, command=import_csv, borderwidth=0, bg="#424242", activebackground="#424242")
reset_button = Button(program_frame, image=reset_img, command=reset, borderwidth=0, bg="#424242", activebackground="#424242")
plot_canvas = Canvas(program_frame, height=500, width=500, bg="white")
exit_button = Button(image=exit_img, command=exit, borderwidth=0, bg="#212121", activebackground="#212121")

# Pack/Grid objects and frames
title.pack(padx=10)
credits.pack(padx=10)
program_frame.pack(pady=5)
exit_button.pack(padx=10)

csv_label.grid(row=3, column=0, padx=10)
input_frame.grid(row=4, column=0, padx=15)

csv_input.grid(row=0, column=0)
browse_button.grid(row=0, column=1)

predict_button.grid(row=5, column=0, padx=10)
csv_status.grid(row=6, column=0, padx=10)
result_label.grid(row=7, column=0, padx=0)
reset_button.grid(row=8, column=0, padx=10)
plot_canvas.grid(row=0, column=1, rowspan=10, padx=25, pady=25)
# Run GUI Program
window.mainloop()