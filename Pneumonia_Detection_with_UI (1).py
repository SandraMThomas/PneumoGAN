import tkinter as tk
from time import *
from tkinter import filedialog
from tkinter import *
from time import sleep
import numpy as np
from PIL import Image, ImageTk # python imaging library : opening, manipulating, etc.
import tkinter.font as font 
from keras.models import load_model
from keras.preprocessing import image




# size = 1200,630

class app_one:
    
    def init(self):
        global screen_width,screen_height
        
        self.window = tk.Tk()
        
        #background image
        self.smec = Image.open("./resources/img/back.jpeg")
        self.backgroundimage = ImageTk.PhotoImage(self.smec)
        self.backgroundimage1= Label(self.window, image= self.backgroundimage)
        self.backgroundimage1.place(x=0, y=0, width=1000, height=630)
    
        #title name
        self.title = Label(master=self.window, text="Pneumonia Detection System", font=(
        "Lucida Calligraphy", 30), fg="white", bg="#3d0099", height=1)
        self.title.place(x=0,y=0)
        self.title.pack()
        
        #output box
        self.Message = Label(self.window,text ="Output",width=23,
                          height=4,font=(
        "Lucida Calligraphy", 30))
        self.Message.place(x=400,y=400)
 

        #upload button
        upload_btn_img = Image.open("./resources/img/Upload.png")
        upload_btn_img = upload_btn_img.resize((200, 200))
        upload_btn_img_inp = ImageTk.PhotoImage(upload_btn_img)        
        self.uploadImageButton= tk.Button(image=upload_btn_img_inp,
         command=self.uploadImage, width=250, height=200, bg='#009090')
        self.uploadImageButton.place(x=50, y=120)


        #input imageframe
        self.imageFrame = Frame(self.window,width=250,height=200)
        self.imageFrame.pack_propagate(0)
        self.imageFrame.place(x=390, y=120)
        self.im = Label(self.imageFrame,text = "Image",width=250,height=200)
        self.im.config(bg = "khaki1")
        self.im.pack(side = "top")

        #exit btn
        exit_btn_img = Image.open("./resources/img/exit.jpg")
        exit_btn_img = exit_btn_img.resize((250, 200))
        exit_btn_img_inp = ImageTk.PhotoImage(exit_btn_img)
        self.exitButton = tk.Button(
        image=exit_btn_img_inp,
        command=self.quitFn,
        width=250,
        height=200,
        bg="#FF0000",
        fg="white",
        )
        self.exitButton.place(x=80, y=400)
        

        #execute btn
        execute_btn_img = Image.open("./resources/img/execute.png")
        execute_btn_img = execute_btn_img.resize((250, 200))
        execute_btn_img_inp = ImageTk.PhotoImage(execute_btn_img)
        self.resultButton = tk.Button(
            image=execute_btn_img_inp,
            command=self.result,
            width=250,
            height=200,  
        )
        self.resultButton.place(x=700, y=120)

        ########## MAin FRAME #########################
        
        
        self.window.minsize(width=1000,height = 630)
        self.window.maxsize(width=1000,height = 630)
        self.window.config(bg = 'black')    
        self.window.mainloop()
    
    

    def uploadImage(self):
        
        global base_file
        print("uploadimage..")
        imageFilename = filedialog.askopenfilename(
            initialdir="./", title="Select file")
        print(imageFilename)
        base_file= imageFilename

        self.temp = Image.open(base_file)
        self.temp = self.temp.resize((150,150))
        self.t = ImageTk.PhotoImage(self.temp)
        self.image = tk.Frame(self.window,width=250,height=200)
        self.image.pack_propagate(0)
        self.image.place(x=390, y=120)
        self.im = tk.Label(self.image,image = self.t,width=250,height=200)
        self.im.pack(side = "top")
        self.window.update()
    
        
    def quitFn(self):
        self.window.update()
        sleep(1)
        exit()

    def result(self):
        print('printing result')
        global base_file
        classifier=load_model('./resources/model/dcdis.h5')
        fileActual=base_file
        test_image = image.load_img(fileActual,target_size = (128, 128))# fileActual is the variable in which the path of the selected file
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = classifier.predict(test_image)
        if result[0][0] == 0:
            prediction = 'Not Affected by Pneumonia'
            print(prediction)
            self.Message=Label(self.window,text ="Not Affected by Pneumonia",width=20,
                          height=4,font=(
        "Lucida Calligraphy", 30))
            self.Message.place(x=400,y=400)
            self.window.update()
        else:
            prediction='Affected by Pneumonia'
            print(prediction)
            self.Message=Label(self.window,text ="Affected by Pneumonia",width=23,
                          height=4,font=("Lucida Calligraphy", 30))
            self.Message.place(x=400,y=400)
            self.window.update()


if __name__ == "__main__":
    a = app_one().init()


