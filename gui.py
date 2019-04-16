from tkinter import *
import driver





def ignite():
    symb = txt.get()
    root.destroy()
    driver.master_process(symb)
    


root = Tk()

lbl1 = Label(root, text = 'Enter Company Symbol')
lbl1.pack()

txt = Entry(root)
txt.pack()

button1 = Button(root,text= 'Predict', command = ignite)
button1.pack()

root.mainloop()