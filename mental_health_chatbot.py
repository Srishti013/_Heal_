import tkinter as tk


def start_mental_chatbot():
    # Clear any existing widgets
    for widget in tk.Toplevel().winfo_children():
        widget.destroy()

    # Create chatbot window
    chatbot_window = tk.Toplevel()
    chatbot_window.title("Mental Health Chatbot")
    chatbot_window.geometry("500x500")
    chatbot_window.config(bg="black")

    # Create label and entry box
    label = tk.Label(chatbot_window, text="How can I assist you today?", fg="white", bg="black", font=("Arial", 16))
    label.pack(pady=20)
    entry = tk.Entry(chatbot_window, font=("Arial", 14))
    entry.pack()

    # Create submit button
    submit_button = tk.Button(chatbot_window, text="Submit", command=lambda: print(entry.get()))
    submit_button.pack(pady=20)
    

