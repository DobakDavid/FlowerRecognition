import tkinter
import tkinter.messagebox
import customtkinter
import threading
import time
import subprocess

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

# Function to run the script
def run_script():
    try:
        # Replace 'python your_script.py' with the command to run your script
        subprocess.run(["python", "main.py"], check=True)
        # main.main()
        print("Script executed successfully!")
    except Exception as e:
        print(f"An error occurred: {e}")

def script_thread():
    t1 = threading.Thread(target = run_script)
    t1.start

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # Configure window
        self.title("Neural network generator GUI")
        self.geometry(f"{1100}x{580}")

        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame, command=self.sidebar_button_1_event, text = "Run script")
        self.sidebar_button_1.grid(row=1, column=0, padx=20, pady=10)

    def sidebar_button_1_event(self):
        print("my button")
        script_thread()

if __name__ == "__main__":
    app = App()
    app.mainloop()
