import tkinter as tk
from PIL import ImageTk, Image
from physical_health_chatbot import start_physical_chatbot
from mental_app import start_mental_health_chat
    

class HEAL:
    def __init__(self, root):
        self.root = root
        self.root.title("HEAL")
        self.root.geometry("500x500")
        self.root.config(bg="black")
        self.welcome_window()
    
    def welcome_window(self):
        # Clear any existing widgets
        for widget in self.root.winfo_children():
            widget.destroy()

        # Create welcome label and entry box
        welcome_label = tk.Label(self.root, text="Welcome to HEAL! Please enter your name:", fg="white", bg="black", font=("Arial", 16))
        welcome_label.pack(pady=30)
        name_entry = tk.Entry(self.root, font=("Arial", 14))
        name_entry.pack()

        # Create submit button
        submit_button = tk.Button(self.root, text="Submit", command=lambda: self.to_chatbot_selection(name_entry.get()))
        submit_button.pack(pady=20)

    def to_chatbot_selection(self, name):
        # Clear any existing widgets
        for widget in self.root.winfo_children():
            widget.destroy()

        # Create label for chatbot selection
        selection_label = tk.Label(self.root, text=f"Hello, {name}! Which chatbot would you like to use?", fg="white", bg="black", font=("Arial", 16))
        selection_label.pack(pady=30)

        # Create buttons for chatbot selection
        mental_button = tk.Button(self.root, text="Mental Health Chatbot", command= start_mental_health_chat)
        physical_button = tk.Button(self.root, text="Physical Health Chatbot", command=start_physical_chatbot)
        mental_button.pack(pady=10)
        physical_button.pack(pady=10)


if __name__ == "__main__":
    root = tk.Tk()
    app = HEAL(root)
    root.mainloop()
