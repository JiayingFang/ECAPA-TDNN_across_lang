import argparse, glob, os, torch, warnings, time
from tools import *
from ECAPAModel import ECAPAModel
import os
import sounddevice as sd
import soundfile

import tkinter as tk
from tkinter import Message, Text
import wave
import csv
from tkinter.messagebox import showinfo
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"


window = tk.Tk()
window.title("Speaker Verification")
window.configure(background='white')
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)
message = tk.Label(
    window, text="Speaker Verification System",
    bg="blue", fg="white", width=55,
    height=3, font=('times', 30, 'bold'))
 
message.place(x=190, y=20)
 
lbl = tk.Label(window, text="No.",
               width=20, height=2, fg="blue",
               bg="white", font=('times', 15, ' bold '))
lbl.place(x=400, y=200)
 
txt = tk.Entry(window,
               width=20, bg="white",
               fg="blue", font=('times', 15, ' bold '))
txt.place(x=700, y=215)
 
lbl2 = tk.Label(window, text="Name",
                width=20, fg="blue", bg="white",
                height=2, font=('times', 15, ' bold '))
lbl2.place(x=400, y=300)
 
txt2 = tk.Entry(window, width=20,
                bg="white", fg="blue",
                font=('times', 15, ' bold '))
txt2.place(x=700, y=315)

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False

def popupmsg(msg):
    popup = tk.Toplevel(window)
    popup.wm_title("!")
    popup.tkraise(window) # This just tells the message to be on top of the root window.
    tk.Label(popup, text=msg).pack(side="top", fill="x", pady=10)
    tk.Button(popup, text="Okay", command = popup.destroy).pack()

parser = argparse.ArgumentParser(description = "ECAPA_trainer")
## Training Settings
parser.add_argument('--num_frames', type=int,   default=200,     help='Duration of the input segments, eg: 200 for 2 second')
parser.add_argument('--max_epoch',  type=int,   default=80,      help='Maximum number of epochs')
parser.add_argument('--batch_size', type=int,   default=160,     help='Batch size')
parser.add_argument('--n_cpu',      type=int,   default=1,       help='Number of loader threads')
parser.add_argument('--save_step',  type=int,   default=1,       help='Test and save every [test_step] epochs')
parser.add_argument('--test_step',  type=int,   default=1,       help='Test and save every [test_step] epochs')
parser.add_argument('--lr',         type=float, default=0.001,   help='Learning rate')
parser.add_argument("--lr_decay",   type=float, default=0.97,    help='Learning rate decay every [test_step] epochs')

## Training and evaluation path/lists, save path
parser.add_argument('--train_list', type=str,   default="/home/ubuntu/user_space/ECAPA-TDNN-main/data/voxceleb2/train_list.txt",     help='The path of the training list, https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/train_list.txt')
parser.add_argument('--train_path', type=str,   default="/home/ubuntu/user_space/ECAPA-TDNN-main/data/voxceleb2/dev/aac",                    help='The path of the training data, eg:"/data08/VoxCeleb2/train/wav" in my case')
parser.add_argument('--target_train_list', type=str,   default="/home/ubuntu/user_space/ECAPA-TDNN-main/data/CN-Celeb_flac/dev/dev.txt",     help='The path of the training list, https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/train_list.txt')
parser.add_argument('--target_train_path', type=str,   default="/home/ubuntu/user_space/ECAPA-TDNN-main/data/CN-Celeb_flac/data",                    help='The path of the training data, eg:"/data08/VoxCeleb2/train/wav" in my case')
parser.add_argument('--eval_list',  type=str,   default="/home/ubuntu/user_space/ECAPA-TDNN-main/data/CN-Celeb_flac/new_test.txt",              help='The path of the evaluation list, veri_test2.txt comes from https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt')
parser.add_argument('--eval_path',  type=str,   default="/home/ubuntu/user_space/ECAPA-TDNN-main/data/CN-Celeb_flac/eval",                    help='The path of the evaluation data, eg:"/data08/VoxCeleb1/test/wav" in my case')
parser.add_argument('--musan_path', type=str,   default="/home/ubuntu/user_space/ECAPA-TDNN-main/data/musan_split",                    help='The path to the MUSAN set, eg:"/data08/Others/musan_split" in my case')
parser.add_argument('--rir_path',   type=str,   default="/home/ubuntu/user_space/ECAPA-TDNN-main/data/RIRS_NOISES/simulated_rirs",     help='The path to the RIR set, eg:"/data08/Others/RIRS_NOISES/simulated_rirs" in my case');
parser.add_argument('--save_path',  type=str,   default="exps/exp1",                                     help='Path to save the score.txt and models')
parser.add_argument('--initial_model',  type=str,   default="",                                          help='Path of the initial_model')

## Model and Loss settings
parser.add_argument('--C',       type=int,   default=1024,   help='Channel size for the speaker encoder')
parser.add_argument('--m',       type=float, default=0.2,    help='Loss margin in AAM softmax')
parser.add_argument('--s',       type=float, default=30,     help='Loss scale in AAM softmax')
parser.add_argument('--n_class', type=int,   default=5994,   help='Number of speakers')

## Command
parser.add_argument('--eval',    dest='eval', action='store_true', help='Only do evaluation')

warnings.simplefilter("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')
args = parser.parse_args()
args = init_args(args)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

s = ECAPAModel(**vars(args))
s.load_parameters("model_0005.model")
s.eval()




def get_enrollment():
    Id = (txt.get())
    name = (txt2.get())
    res = "Recording for : " + Id + " ; Name : " + name
    message.configure(text=res)
    if(is_number(Id) and name.isalpha()):

        showinfo("Message", "Once you press OK, the system will start recording.")
        fs=16000
        duration = 5  # seconds
        myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        showinfo("Message", "Done recording.")
        
        WAVE_OUTPUT_FILENAME = "Enrollment/"+name + "_"+Id + ".wav"

        soundfile.write(WAVE_OUTPUT_FILENAME, myrecording, fs)

        res = "Finish Recording for : " + Id + " ; Name : " + name
        # row = [Id, name]
        # with open('UserDetails\UserDetails.csv', 'a+') as csvFile:
        #     writer = csv.writer(csvFile)
        #     # Entry of the row in csv file
        #     writer.writerow(row)
        # csvFile.close()
        message.configure(text=res)
    else:
        if(is_number(Id)):
            res = "Enter Alphabetical Name"
            message.configure(text=res)
        if(name.isalpha()):
            res = "Enter Numeric Id"
            message.configure(text=res)

def get_testing():
    Id = (txt.get())
    name = (txt2.get())
    res = "Recording for : " + Id + " ; Name : " + name
    message.configure(text=res)
    if(is_number(Id) and name.isalpha()):
        audioPath = "Enrollment/" + name+"_"+Id + ".wav"
        showinfo("Message", "Once you press OK, the system will start recording.")
        fs=16000
        duration = 5  # seconds
        myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        showinfo("Message", "Recording done. The verification is under processing, please wait.")
        WAVE_OUTPUT_FILENAME = "Testing/"+name + "_"+Id + ".wav"

        res = "Finish Recoring for : " + Id + " ; Name : " + name
        message.configure(text=res)

        soundfile.write(WAVE_OUTPUT_FILENAME, myrecording, fs)


        audio, _  = soundfile.read(audioPath)
        data_1 = torch.FloatTensor(numpy.stack([audio],axis=0)).to(device)
        max_audio = 300 * 160 + 240
        if audio.shape[0] <= max_audio:
            shortage = max_audio - audio.shape[0]
            audio = numpy.pad(audio, (0, shortage), 'wrap')
        feats = []
        startframe = numpy.linspace(0, audio.shape[0]-max_audio, num=5)
        for asf in startframe:
            feats.append(audio[int(asf):int(asf)+max_audio])
        feats = numpy.stack(feats, axis = 0).astype(numpy.float)
        data_2 = torch.FloatTensor(feats).to(device)
        with torch.no_grad():
            # embedding_1, sg1 = s.speaker_encoder.forward(data_1, aug = False)
            # embedding_1 = F.normalize(embedding_1, p=2, dim=1)
            embedding_2, sg2 = s.speaker_encoder.forward(data_2, aug = False)
            embedding_2 = F.normalize(embedding_2, p=2, dim=1)

        max_audio = 300 * 160 + 240
        audio_test1, _  = soundfile.read(WAVE_OUTPUT_FILENAME)
        data_test1_1 = torch.FloatTensor(numpy.stack([audio_test1],axis=0)).to(device)
        if audio_test1.shape[0] <= max_audio:
            shortage = max_audio - audio_test1.shape[0]
            audio_test1= numpy.pad(audio_test1, (0, shortage), 'wrap')
        feats_test1 = []
        startframe_test1 = numpy.linspace(0, audio_test1.shape[0]-max_audio, num=5)
        for asf in startframe_test1:
            feats_test1.append(audio_test1[int(asf):int(asf)+max_audio])
        feats_test1 = numpy.stack(feats_test1, axis = 0).astype(numpy.float)
        data_test1_2 = torch.FloatTensor(feats_test1).to(device)

        with torch.no_grad():
            # embedding_test1_1, sg1 = s.speaker_encoder.forward(data_test1_1, aug = False)
            # embedding_test1_1 = F.normalize(embedding_test1_1, p=2, dim=1)
            embedding_test1_2, sg2 = s.speaker_encoder.forward(data_test1_2, aug = False)
            embedding_test1_2 = F.normalize(embedding_test1_2, p=2, dim=1)

        # score_1 = torch.mean(torch.matmul(embedding_1, embedding_test1_1.T))
        score_2 = torch.mean(torch.matmul(embedding_2, embedding_test1_2.T))
        # score = (score_1 + score_2) / 2
        score = score_2
        score = score.detach().cpu().numpy()
        print(score)
        showinfo("Message", "The comparison score is: " + str(score))

        
        # popupmsg("The comparison score is: {score}.")
        if score > 0.3:
            res = "Yes, this is the same speaker."
            message.configure(text=res)
        else:
            res = "No, this is not the same speaker."
            message.configure(text=res)
    else:
        if(is_number(Id)):
            res = "Enter Alphabetical Name"
            message.configure(text=res)
        if(name.isalpha()):
            res = "Enter Numeric Id"
            message.configure(text=res)

takeEnroll = tk.Button(window, text="Register",
                    command=get_enrollment, fg="black", bg="green",
                    width=20, height=3, activebackground="Red",
                    font=('times', 15, ' bold '))
takeEnroll.place(x=200, y=500)
takeVeri = tk.Button(window, text="Verification",
                     command=get_testing, fg="black", bg="green",
                     width=20, height=3, activebackground="Red",
                     font=('times', 15, ' bold '))
takeVeri.place(x=500, y=500)

quitWindow = tk.Button(window, text="Quit",
                       command=window.destroy, fg="black", bg="green",
                       width=20, height=3, activebackground="Red",
                       font=('times', 15, ' bold '))
quitWindow.place(x=800, y=500)
 
 
window.mainloop()