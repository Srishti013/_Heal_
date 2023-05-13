import tkinter as tk


def start_physical_chatbot():
    # Clear any existing widgets
    for widget in tk.Toplevel().winfo_children():
        widget.destroy()

    # Create chatbot window
    chatbot_window = tk.Toplevel()
    chatbot_window.title("Physical Health Chatbot")
    chatbot_window.geometry("500x500")
    chatbot_window.config(bg="black")

    # Create label and entry box
    label = tk.Label(chatbot_window, text="How can I assist you today?", fg="white", bg="black")
