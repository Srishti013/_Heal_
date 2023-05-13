import tkinter as tk
from tkinter import *
from physical import *
from tkinter import simpledialog

bot_name = "HEAL"

BG_GRAY = "#ABB2B9"
BG_COLOR = "black"
TEXT_COLOR = "white"

FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"

def start_physical_health_chat():
        app = ChatApplication()
        app.run()

class ChatApplication:
    
    def __init__(self):
        self.window = Tk()
        self._setup_main_window()

    
    def run(self):
        self.window.mainloop()
        
    def _setup_main_window(self):
        self.window.title("Chat")
        self.window.resizable(width=False, height=False)
        self.window.configure(width=500, height=550, bg=BG_COLOR)
        
        # head label
        head_label = Label(self.window, bg=BG_COLOR, fg=TEXT_COLOR,
                           text="Welcome to HEAL", font=FONT_BOLD, pady=10)
        head_label.place(relwidth=1)

        # tiny divider
        line = Label(self.window, width=450, bg=BG_GRAY)
        line.place(relwidth=1, rely=0.07, relheight=0.012)
        
        # text widget
        self.text_widget = Text(self.window, width=20, height=2, bg=BG_COLOR, fg=TEXT_COLOR,
                                font=FONT, padx=5, pady=5)
        self.text_widget.place(relheight=0.745, relwidth=1, rely=0.08)
        self.text_widget.configure(cursor="arrow", state=NORMAL)

        # scroll bar
        scrollbar = Scrollbar(self.text_widget)
        scrollbar.place(relheight=1, relx=0.974)
        scrollbar.configure(command=self.text_widget.yview)
        
        # bottom label
        bottom_label = Label(self.window, bg=BG_GRAY, height=80)
        bottom_label.place(relwidth=1, rely=0.825)
        
        # message entry box
        self.msg_entry = Entry(bottom_label, bg="#2C3E50", fg=TEXT_COLOR, font=FONT)
        self.msg_entry.place(relwidth=0.74, relheight=0.06, rely=0.008, relx=0.011)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>", self._on_enter_pressed)
        
        # send button
        send_button = Button(bottom_label, text="Send", font=FONT_BOLD, width=20, bg=BG_GRAY,
                             command=lambda: self._on_enter_pressed(None))
        send_button.place(relx=0.77, rely=0.008, relheight=0.06, relwidth=0.22)
        
        
        
        
        self.text_widget.insert(tk.END, f"{bot_name}: Hi Welcome to HEAL, Enter the main symptom you are experiencing\n\n")
        sym1 = self.msg_entry.get()
        sym1 = preprocess_sym(sym1)
        sim1, psym1 = syntactic_similarity(sym1, all_symp_pr)
        print(f"sym1: {sym1}")
        if sim1 == 1:
            psym1 = related_sym(psym1)
        self.text_widget.insert(tk.END, f"{bot_name}:  Enter a second symptom you are experiencing,\n\n")
        sym2 = self.msg_entry.get()
        print(f"sym2: {sym2}")
        self.msg_entry.delete(0, tk.END)
        sym2=preprocess_sym(sym2)
        sim2,psym2=syntactic_similarity(sym2,all_symp_pr)
        if sim2==1:
            psym2=related_sym(psym2)
         #if check_pattern==0 no similar syntaxic symp1 or symp2 ->> try semantic similarity
    
        if sim1==0 or sim2==0:
            sim1,psym1=semantic_similarity(sym1,all_symp_pr)
            sim2,psym2=semantic_similarity(sym2,all_symp_pr)
        
        #if semantic sim syp1 ==0 (no symp found) ->> suggest possible data symptoms based on all data and input sym synonymes
        if sim1==0:
            sugg=suggest_syn(sym1)
            self.text_widget.insert(tk.END, f"{bot_name}:  Are you experiencing any\n\n")
            for res in sugg:
                self.text_widget.insert(tk.END, f"{bot_name}:  {res}?\n\n")
                inp = self.msg_entry.get()
                self.msg_entry.delete(0, tk.END)
                if inp=="yes":
                    psym1=res
                    sim1=1
                    break
                
        #if semantic sim syp2 ==0 (no symp found) ->> suggest possible data symptoms based on all data and input sym synonymes
        if sim2==0:
            sugg=suggest_syn(sym2)
            for res in sugg:
                self.text_widget.insert(tk.END, f"{bot_name}:  Do you feel {res} ?(yes or no) \n\n")
                inp = self.msg_entry.get()
                self.msg_entry.delete(0, tk.END)
                if inp=="yes":
                    psym2=res
                    sim2=1
                    break
        #if no syntaxic semantic and suggested sym found return None and ask for clarification

        if sim1==0 and sim2==0:
            return None,None
        else:
            # if at least one sym found ->> duplicate it and proceed
            if sim1==0:
                psym1=psym2
            if sim2==0:
                psym2=psym1
        #create patient symp list
        all_sym=[col_dict[psym1],col_dict[psym2]]
        #predict possible diseases
        diseases=possible_diseases(all_sym)
        stop=False
    
        self.text_widget.insert(tk.END, f"{bot_name}: Are you experiencing any\n")
        for dis in diseases:
            if stop==False:
                for sym in symVONdisease(df_tr,dis):
                    if sym not in all_sym:
                        self.text_widget.insert(tk.END, f"{bot_name}: {clean_symp(sym)}+' ?'\n\n")
                        while True:
                            inp = self.msg_entry.get()
                            self.msg_entry.delete(0, tk.END)
                            if(inp=="yes" or inp=="no"):
                                break
                            else:
                                self.text_widget.insert(tk.END, f"{bot_name}: provide proper answers i.e. (yes/no) : \n\n")
                        if inp=="yes":
                            all_sym.append(sym)
                            diseases=possible_diseases(all_sym)
                            if len(diseases)==1:
                                stop=True 
        
        result,sym = knn_clf.predict(OHV(all_sym,all_symp_col)),all_sym
        
        if result == None :
            self.text_widget.insert(tk.END, f"{bot_name}: can you specify more what you feel or tap q to stop the conversation\n\n")
            ans3 = self.msg_entry.get()
            self.msg_entry.delete(0, tk.END)
            if ans3=="q":
                return
        
        else:
            self.text_widget.insert(tk.END,f"{bot_name}: you may have {result[0]}\n\n" )
            self.text_widget.insert(tk.END, f"{bot_name}: {description_list[result[0]]}\n\n")
            self.text_widget.insert(tk.END, f"{bot_name}: how many day do you feel those symptoms ?\n\n")
            an = self.msg_entry.get()
            self.msg_entry.delete(0, tk.END)
            if calc_condition(sym,int(an))==1:
                self.text_widget.insert(tk.END, f"{bot_name}: you should take the consultation from doctor\n\n")
            else : 
                self.text_widget.insert(tk.END, f"{bot_name}: Take following precautions : \n")
                for e in precautionDictionary[result[0]]:
                    self.text_widget.insert(tk.END, f"{bot_name}: {e}\n")
            self.text_widget.insert(tk.END, f"\n{bot_name}: do you need another medical consultation (yes or no)?\n\n")
            ans=self.msg_entry.get()
            self.msg_entry.delete(0, tk.END)
            if ans!="yes":
                self.text_widget.insert(tk.END, f"\n{bot_name}: !!!!! THANKS FOR YOUR VISIT :) !!!!!! \n\n")
                return

        
     
    def _on_enter_pressed(self, event):
        msg = self.msg_entry.get()
        self._insert_message(msg, "You")
        
    def _insert_message(self, msg, sender):
        if not msg:
            return
        
        self.msg_entry.delete(0, END)
        msg1 = f"{sender}: {msg}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg1)
        self.text_widget.configure(state=DISABLED)
        
        msg2 = f"{bot_name}: {get_response(msg)}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg2)
        self.text_widget.configure(state=DISABLED)
        
        self.text_widget.see(END)
        
if __name__ == "__main__":
    app = ChatApplication()
    app.run()