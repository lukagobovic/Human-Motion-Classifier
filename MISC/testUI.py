import tkinter as tk
import tkinter.messagebox
import customtkinter as ttk
import os
from tkinter import Tk, Label, Entry, Button, filedialog, StringVar, W
from tkinter import filedialog
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pickle
from CTkMessagebox import CTkMessagebox
from scipy.stats import skew, kurtosis



ttk.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
ttk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

window_size = 5
# Define the preprocessing function
def preprocess_data(data):
  # Extract features
  features = [
    np.max(data.x),
    np.min(data.x),
    np.ptp(data.x),
    np.mean(data.x),
    np.median(data.x),
    skew(data.x),
    np.var(data.x),
    np.std(data.x),
    kurtosis(data.x),
    np.sqrt(np.mean(data.x ** 2)),
    np.max(data.y),
    np.min(data.y),
    np.ptp(data.y),
    np.mean(data.y),
    np.median(data.y),
    skew(data.y),
    np.var(data.y),
    np.std(data.y),
    kurtosis(data.y),
    np.sqrt(np.mean(data.y ** 2)),
    np.max(data.z),
    np.min(data.z),
    np.ptp(data.z),
    np.mean(data.z),
    np.median(data.z),
    skew(data.z),
    np.var(data.z),
    np.std(data.z),
    kurtosis(data.z),
    np.sqrt(np.mean(data.z ** 2)),
    np.max(data.total_acceleration),
    np.min(data.total_acceleration),
    np.ptp(data.total_acceleration),
    np.mean(data.total_acceleration),
    np.median(data.total_acceleration),
    skew(data.total_acceleration),
    np.var(data.total_acceleration),
    np.std(data.total_acceleration),
    kurtosis(data.total_acceleration),
    np.sqrt(np.mean(data.total_acceleration ** 2))
  ]

  return features

  # Show a message box with the path to the output file
  CTkMessagebox.showinfo("Success", "Predictions saved to {}".format(output_path))

class WelcomeScreen(ttk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.master.geometry("500x400")
        self.pack()
        self.create_widgets()  
        
    def create_widgets(self):
        # window_height = 500
        # window_width = 400      

        # screen_width = self.winfo_screenwidth()
        # screen_height = self.winfo_screenheight()

        # x_cordinate = int((screen_width/2) - (window_width/2))
        # y_cordinate = int((screen_height/2) - (window_height/2)) 
        
        welcome_label = ttk.CTkLabel(self, text="Welcome to the App!",width=400,height=200,corner_radius=8)
        welcome_label.pack(padx=50,pady = 10)
        
        start_button = ttk.CTkButton(self, text="Start", command=self.start_app)
        start_button.pack(pady=10)
        
    def start_app(self):
        self.master.destroy()
        root = ttk.CTk()
        app = App()
        app.mainloop()



class App(ttk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Human Motion Classifier")

        self.input_file_label = ttk.CTkLabel(self, text="Input File:", font=("Segoe UI", 14))
        self.input_file_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        self.input_file_path = tk.StringVar()
        self.input_file_entry = ttk.CTkEntry(self, textvariable=self.input_file_path, width=500, font=("Segoe UI", 12))
        self.input_file_entry.grid(row=0, column=1, padx=10, pady=10)

        self.browse_button = ttk.CTkButton(self, text="Browse", font=("Segoe UI", 12), command=self.browse)
        self.browse_button.grid(row=0, column=2, padx=10, pady=10)

        self.predict_button = ttk.CTkButton(self, text="Predict", font=("Segoe UI", 12), command=self.predict)
        self.predict_button.grid(row=1, column=1, padx=10, pady=10)

        self.status_label = ttk.CTkLabel(self, text="", font=("Segoe UI", 14))
        self.status_label.grid(row=2, column=0, padx=10, pady=10, sticky="w")

        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, self)
        self.canvas.get_tk_widget().configure(background='black')
        self.canvas.get_tk_widget().grid(row=3, column=0, columnspan=3, padx=10, pady=10)

        self.protocol("WM_DELETE_WINDOW", self.on_closing)


    def on_closing(self):
        if tkinter.messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.destroy()
            exit()

         # set protocol to call on_closing() function when window is closed

    def browse(self):
        file_path = filedialog.askopenfilename()
        self.input_file_entry.delete(0, tk.END)
        self.input_file_entry.insert(0, file_path)

    def browse_output_file(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        self.output_file_entry.insert(0, file_path)

    def predict(self):
        input_file_path = self.input_file_entry.get()
        #output_file_path = self.output_file_entry.get()

        # Load data
        originalData = pd.read_csv(input_file_path, delimiter=',')
        # Remove outliers
        df = originalData.iloc[:,1:]
        Q1 = df.quantile(0.20)
        Q3 = df.quantile(0.80)
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

        # Load the feature data

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
                segment.rename(columns={0: "x", 1: "y", 2: "z", 3: "total_acceleration"}, inplace=True)
                segments.append(segment)
            return segments

        # Segment the data and shuffle it
        segments = segment_data(originalData, window_size)
        np.random.shuffle(segments)
        

        train_features = [preprocess_data(segment) for segment in segments]

        # Predict labels
        y_pred = model.predict(train_features)

        # Make predictions
        predicted_class = np.bincount(y_pred.astype(int)).argmax()
        if predicted_class == 0:
            print("This CSV contains walking data.")
            data['Walking(0)/Jumping(1)'] = 0
            originalData.to_csv('OutputData/'+os.path.basename(input_file_path), index=False)
        else:
            print("This CSV contains jumping data.")
            data['Walking(0)/Jumping(1)'] = 1
            originalData.to_csv('OutputData/'+os.path.basename(input_file_path), index=False)

        # Write output file
        # with open(output_file_path, 'w') as f:
        #     f.write('label\n')
        #     for label in labels:
        #         f.write(label + '\n')

        # Plot predictions
        ax = self.fig.add_subplot(111)
        ax.plot(data.iloc[0:1000,1:5])
        ax.set_xlabel('Window')
        ax.set_ylabel('Probability')
        #ax.legend(['walking', 'jumping'], loc='upper right')
        self.canvas.draw()

welcome_root = ttk.CTk()
welcome_screen = WelcomeScreen(welcome_root)
welcome_root.mainloop()