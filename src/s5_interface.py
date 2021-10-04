if True:  # Include project path
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)

    import matplotlib.pyplot as plt
    import numpy as np
    import cv2
    import argparse
    import utils.lib_images_io as lib_images_io
    import utils.lib_plot as lib_plot
    import utils.lib_commons as lib_commons
    from utils.lib_openpose import SkeletonDetector
    from utils.lib_tracker import Tracker
    from utils.lib_classifier import *  # Import all sklearn related libraries
    from utils.lib_feature_proc import FeatureGenerator
    from tensorflow.keras.models import load_model
    
    from tkinter import *
    from PIL import ImageTk, Image
    import time
    import random
    import yaml
    from datetime import date
    import csv



def par(path):  # Pre-Append ROOT to the path if it's not absolute
    return ROOT + path if (path and path[0] != "/") else path


# -- Command-line input


def get_command_line_arguments():

    def parse_args():
        parser = argparse.ArgumentParser(
            description="Test action recognition on \n"
            "(1) a video, (2) a folder of images, (3) or web camera.")
        parser.add_argument("-m", "--model_path", required=False,
                            default='model/trained_classifier.pickle')
        parser.add_argument("-t", "--data_type", required=False, default='webcam',
                            choices=["video", "folder", "webcam"])
        parser.add_argument("-p", "--data_path", required=False, default="",
                            help="path to a video file, or images folder, or webcam. \n"
                            "For video and folder, the path should be "
                            "absolute or relative to this project's root. "
                            "For webcam, either input an index or device name. ")
        parser.add_argument("-o", "--output_folder", required=False, default='output/',
                            help="Which folder to save result to.")

        args = parser.parse_args()
        return args
    args = parse_args()
    if args.data_type != "webcam" and args.data_path and args.data_path[0] != "/":
        # If the path is not absolute, then its relative to the ROOT.
        args.data_path = ROOT + args.data_path
    return args


def get_dst_folder_name(src_data_type, src_data_path):
    ''' Compute a output folder name based on data_type and data_path.
        The final output of this script looks like this:
            DST_FOLDER/folder_name/video.avi
            DST_FOLDER/folder_name/skeletons/XXXXX.txt
    '''

    assert(src_data_type in ["video", "folder", "webcam"])

    if src_data_type == "video":  # /root/data/video.avi --> video
        folder_name = os.path.basename(src_data_path).split(".")[-2]

    elif src_data_type == "folder":  # /root/data/video/ --> video
        folder_name = src_data_path.rstrip("/").split("/")[-1]

    elif src_data_type == "webcam":
        # month-day-hour-minute-seconds, e.g.: 02-26-15-51-12
        folder_name = lib_commons.get_time_string()

    return folder_name


args = get_command_line_arguments()

SRC_DATA_TYPE = args.data_type
SRC_DATA_PATH = args.data_path
SRC_MODEL_PATH = args.model_path

# -- Settings

cfg_all = lib_commons.read_yaml(ROOT + "config/config.yaml")
cfg = cfg_all["s5_interface.py"]

classes = np.array(cfg_all["classes"])

badges_path = cfg["input"]["badges"]
user_data_path = cfg["input"]["data_path"]

# Action recognition: number of frames used to extract features.
WINDOW_SIZE = int(cfg_all["features"]["window_size"])

# Video settings

# If data_type is webcam, set the max frame rate.
SRC_WEBCAM_MAX_FPS = float(cfg["settings"]["source"]
                           ["webcam_max_framerate"])

# If data_type is video, set the sampling interval.
# For example, if it's 3, then the video will be read 3 times faster.
SRC_VIDEO_SAMPLE_INTERVAL = int(cfg["settings"]["source"]
                                ["video_sample_interval"])

# Openpose settings
OPENPOSE_MODEL = cfg["settings"]["openpose"]["model"]
OPENPOSE_IMG_SIZE = cfg["settings"]["openpose"]["img_size"]

# -- Function

def select_images_loader(src_data_type, src_data_path):
    if src_data_type == "video":
        images_loader = lib_images_io.ReadFromVideo(
            src_data_path,
            sample_interval=SRC_VIDEO_SAMPLE_INTERVAL)

    elif src_data_type == "folder":
        images_loader = lib_images_io.ReadFromFolder(
            folder_path=src_data_path)

    elif src_data_type == "webcam":
        if src_data_path == "":
            webcam_idx = 0
        elif src_data_path.isdigit():
            webcam_idx = int(src_data_path)
        else:
            webcam_idx = src_data_path
        images_loader = lib_images_io.ReadFromWebcam(
            SRC_WEBCAM_MAX_FPS, webcam_idx)
    return images_loader

class MultiPersonClassifier(object):
    ''' This is a wrapper around ClassifierOnlineTest
        for recognizing actions of multiple people.
    '''

    def __init__(self, model_path, classes):

        self.dict_id2clf = {}  # human id -> classifier of this person

        # Define a function for creating classifier for new people.
        self._create_classifier = lambda human_id: ClassifierOnlineTest(
            model_path, classes, WINDOW_SIZE)

    def classify(self, dict_id2skeleton):
        ''' Classify the action type of each skeleton in dict_id2skeleton '''

        # Clear people not in view
        old_ids = set(self.dict_id2clf)
        cur_ids = set(dict_id2skeleton)
        humans_not_in_view = list(old_ids - cur_ids)
        for human in humans_not_in_view:
            del self.dict_id2clf[human]

        # Predict each person's action
        id2label = {}
        for id, skeleton in dict_id2skeleton.items():

            if id not in self.dict_id2clf:  # add this new person
                self.dict_id2clf[id] = self._create_classifier(id)

            classifier = self.dict_id2clf[id]
            id2label[id] = classifier.predict(skeleton)  # predict label
            # print("\n\nPredicting label for human{}".format(id))
            # print("  skeleton: {}".format(skeleton))
            # print("  label: {}".format(id2label[id]))

        return id2label

    def get_classifier(self, id):
        #Get the classifier based on the person id.
        #Arguments:
        #   id {int or "min"}

        if len(self.dict_id2clf) == 0:
            return None
        if id == 'min':
            id = min(self.dict_id2clf.keys())
        return self.dict_id2clf[id]


def remove_skeletons_with_few_joints(skeletons):
    ''' Remove bad skeletons before sending to the tracker '''
    good_skeletons = []
    for skeleton in skeletons:
        px = skeleton[2:2+13*2:2]
        py = skeleton[3:2+13*2:2]
        num_valid_joints = len([x for x in px if x != 0])
        num_leg_joints = len([x for x in px[-6:] if x != 0])
        total_size = max(py) - min(py)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # IF JOINTS ARE MISSING, TRY CHANGING THESE VALUES:
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if num_valid_joints >= 5 and total_size >= 0.1 and num_leg_joints >= 0:
            # add this skeleton only when all requirements are satisfied
            good_skeletons.append(skeleton)
    return good_skeletons

def get_resized_image(filepath, base_width):
    img = Image.open(filepath)
    basewidth = base_width
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    #print("H " + str(hsize))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    pi_img = ImageTk.PhotoImage(img)
    return pi_img

def get_joints(class_idx, left = False):
    legs = []
    foldedlegs = []
    if class_idx == 3:
        if(left == True):
            legs = [8,9,10]
        else:
            legs = [11,12,13]
    elif class_idx == 4:
        if(left == True):
            foldedlegs = [9,8,12] #[10,9,12]
        else:
            foldedlegs = [12,11,9] #[13,12,9]
    switcher = {
        0: [11,5,6], #jumpingjack
        1: [9,11,12], #squat
        2: [11,5,6], #arms
        3: legs, #legs
        4: foldedlegs, #foldedlegs
        5: [3,0,6] #[5,7,6] #shoulders
    }
    return switcher.get(class_idx, "Invalid class")

def calculate_angle(skeleton, class_idx):

    if(class_idx == -1):
        first,second,third = 11, 5, 7 #stand
    else:
        first, second, third = get_joints(class_idx)[0], get_joints(class_idx)[1], get_joints(class_idx)[2]
    
    radians = np.arctan2(skeleton[2*third+1]-skeleton[2*second+1], 
        skeleton[2*third]-skeleton[2*second])- np.arctan2(skeleton[2*first+1]-skeleton[2*second+1],
        skeleton[2*first]-skeleton[2*second])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
    
    if(class_idx == 3 or class_idx == 4):
        first, second, third = get_joints(class_idx,True)[0], get_joints(class_idx,True)[1], get_joints(class_idx,True)[2]
        
        radians = np.arctan2(skeleton[2*third+1]-skeleton[2*second+1], 
            skeleton[2*third]-skeleton[2*second])- np.arctan2(skeleton[2*first+1]-skeleton[2*second+1],
            skeleton[2*first]-skeleton[2*second])
        angle_sx = np.abs(radians*180.0/np.pi)
        
        if angle_sx >180.0:
            angle_sx = 360-angle_sx
        
        if(angle_sx > angle):
            angle = angle_sx

    return float(format(angle, ".2f"))

def update_badges():
    global badges
    global badge_class
    global badge_medal
    badges_doc = []
    found = False
    if (num_reps != 0):
        with open(badges_path, 'r') as file:
            badges_doc = yaml.load(file, Loader=yaml.FullLoader)

            for idx_dict in range(len(badges_doc)):
                keys = list(badges_doc[idx_dict].keys())
                num_class = str("n_"+classes[class_idx])
                if(keys[0] == num_class):
                    values = list(badges_doc[idx_dict].values())
                    badges_doc[idx_dict][num_class] = values[0] + num_reps
                    break

            for idx_exercise in range(len(badges_doc)):
                if badges_doc[idx_exercise].get(classes[class_idx],"error") != "error":
                    if(badges_doc[idx_dict][num_class] >= 100 and 
                    badges_doc[idx_exercise][classes[class_idx]] == "none"):
                        badges_doc[idx_exercise][classes[class_idx]] = badge_medal ="bronzo"
                        badge_class = class_idx
                    elif(badges_doc[idx_dict][num_class] >= 200 and 
                    badges_doc[idx_exercise][classes[class_idx]] == "bronzo"):
                        badges_doc[idx_exercise][classes[class_idx]] = badge_medal ="argento"
                        badge_class = class_idx
                    elif(badges_doc[idx_dict][num_class] >= 300 and 
                    badges_doc[idx_exercise][classes[class_idx]] == "argento"):
                        badges_doc[idx_exercise][classes[class_idx]] = badge_medal ="oro"
                        badge_class = class_idx
                    break
            
            for index in range(len(badges_doc)):
                keys = list(badges_doc[index].keys())
                if (keys[0][0:2] != "n_"):
                    values = list(badges_doc[index].values())
                    badges[keys[0]] = values[0]

        if len(badges_doc) != 0:    
            with open(badges_path,'w') as file:
                yaml.dump(badges, file)

def write_csv():
    global user_data
    with open(user_data_path, 'w',newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(user_data)
    user_data = []

def load_csv():
    with open(user_data_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        history = []
        for row in csv_reader:
            history.append(row)
    return history

def congratulations():
    global congratulazioni
    global loop
    congratulazioni = True
        
    root.after_cancel(loop)
    class_lbl.place_forget()
    reps_lbl.place_forget()
    text_lbl.place_forget()
    camera_lbl.place_forget()
    button_concludi.place_forget()
    #319

    no_badge = ""
    if(badge_medal == "none"):
        no_badge = "_no_badge"
    else:
        img_badge = get_resized_image(str("src/images/" + classes[badge_class] + "_" + badge_medal + ".png"),250)
        lbl_badge.configure(image=img_badge)
        lbl_badge.imgtk = img_badge
        lbl_badge.place(x=280, y=335)

    img_congratulations = get_resized_image(str("src/images/congratulazioni" + no_badge +".png"),800)
    screen.configure(image=img_congratulations)
    screen.imgtk = img_congratulations

    button_avanti.place(x=290, y = 410)
    loop = root.after(1,congratulations)

def feedback():
    #... gestione del feedback
    global fb
    fb = True
    update_badges()
    profile_or_workout()

def terminate_workout():
    root.after_cancel(loop)
    class_lbl.place_forget()
    reps_lbl.place_forget()
    text_lbl.place_forget()
    camera_lbl.place_forget()
    button_concludi.place_forget()

    global user_data
    user_data.append(num_reps)
    if len(user_data) != 7:
        for i in range(7 - len(user_data)):
            user_data.append(0)
    
    #319

    terminate_image = get_resized_image("src/images/screen_terminate.png",800)
    screen.configure(image=terminate_image)
    screen.imgtk = terminate_image

    button_hard_ex.place(x = 160, y = 265)
    button_dontlike_ex.place(x=410, y = 265)
    button_tired.place(x=160, y = 335)
    button_dont_say.place(x=410, y = 335)


def popup():
    global popup

    popup = Toplevel(root)
    popup.lift(root)
    popup.title("Gym AI Assistant")
    popup.geometry("500x312")
    #popup.resizable(width=False, height=False)

    img_screen_popup = get_resized_image('src/images/popup.png', 500)
    screen_popup = Label(popup, image=img_screen_popup, borderwidth=0)
    screen_popup.image = img_screen_popup
    screen_popup.place(x = 0,y = 0)

    button_pop_continue = Button(popup, image=img_pop_continue,
    bg="#343434",command=popup.destroy,activebackground='white', borderwidth=0)

    button_pop_stop = Button(popup, image=img_pop_stop,
    bg="#343434",command=feedback ,activebackground='white', borderwidth=0)

    button_pop_continue.place(x = 25, y = 250)
    button_pop_stop.place(x = 260, y = 250)

'''
if len(dict_id2skeleton):
    classifier_of_a_person = multiperson_classifier.get_classifier(id='min')
    scores = classifier_of_a_person.get_scores()
    #... ( per l'accuratezza dell'esercizio )
'''

def max_angle(class_idx):
    switcher = {
        0: 120, #jumpingjack
        1: 70, #squat
        2: 90, #arms
        3: 120, #legs
        4: 60, #foldedlegs
        5: 130 #shoulders
    }
    return switcher.get(class_idx, "Invalid class")

def min_angle(class_idx):
    switcher = {
        0: 18, #jumpingjack
        1: 45, #squat
        2: 60, #arms
        3: 100, #legs
        4: 35, #foldedlegs
        5: 90 #shoulders
    }
    return switcher.get(class_idx, "Invalid class")



# function for video streaming
def video_stream():
    global num_frame
    global interval
    global last_motivation_number
    global class_idx
    global num_reps
    global angle
    global start_position
    global negative_phase
    global reached_angle
    global loop
    global user_data

    if(num_reps == tot_reps):
        user_data.append(num_reps)
        update_badges()
        class_idx = class_idx + 1
        if class_idx == len(classes):
            num_reps = 0
            class_idx = 0
            congratulations()
        else:           
            num_reps = 0
            angle = 0
            img_class = get_resized_image("src/images/" + str(classes[class_idx]) + ".png", 170)
            class_lbl.configure(image=img_class)
            class_lbl.imgtk = img_class
            start_position = False

    #prendo l'immagine, calcolo lo scheletro
    camera = images_loader.read_image()

    # -- Detect skeletons
    humans = skeleton_detector.detect(camera)
    skeletons, scale_h = skeleton_detector.humans_to_skels_list(humans)
    skeletons = remove_skeletons_with_few_joints(skeletons)

    # -- Track people -- int id -> np.array() skeleton
    dict_id2skeleton = multiperson_tracker.track(skeletons)

    # -- Recognize action of each person
    if len(dict_id2skeleton):
        dict_id2label = multiperson_classifier.classify(dict_id2skeleton)
        skeleton = skeletons[0]
        angle_stand = calculate_angle(skeleton, -1)
        if (angle_stand < 45):
            start_position = True
        if start_position == True:
            new_angle = calculate_angle(skeleton, class_idx)
            if (new_angle > max_angle(class_idx) and 
                    new_angle <= max_angle(class_idx) + 20):
                reached_angle = True
            if(new_angle < min_angle(class_idx)):
                negative_phase = True
            elif(angle!=0 and negative_phase == True and reached_angle == True):
                num_reps = num_reps + 1
                reached_angle = False
                negative_phase = False
            angle = new_angle
        if(num_frame == interval):
            classifier_of_a_person = multiperson_classifier.get_classifier(id='min')
            score = classifier_of_a_person.get_score(class_idx)
            if(score == 0.0):
                popup()

    #draw skeleton joints
    skeleton_detector.draw(camera, humans)
    #image resize
    cv2image = cv2.cvtColor(camera, cv2.COLOR_BGR2RGBA)
    camera = Image.fromarray(cv2image)
    basewidth = 560
    hsize = int(camera.size[1]*basewidth/camera.size[0])
    camera = camera.resize((basewidth, hsize), Image.ANTIALIAS)

    pi_camera = ImageTk.PhotoImage(image=camera)
    camera_lbl.imgtk = pi_camera
    camera_lbl.configure(image=pi_camera)
    camera_lbl.place(x=220, y=10)

    reps_lbl.configure(text=str(num_reps)+"/"+str(tot_reps))
    reps_lbl.place(x = 24, y = 230)

    button_concludi.place(x=20, y=397)

    if(num_frame == interval):
        n_motivation = last_motivation_number
        while (n_motivation == last_motivation_number):
            n_motivation = random.randint(1,3)
        last_motivation_number = n_motivation
        img_motivation = get_resized_image("src/images/motivation_" + str(n_motivation) + ".png", 350)
        text_lbl.configure(image=img_motivation)
        text_lbl.imgtk = img_motivation
    elif (num_frame == (interval + 250)):
        text_lbl.configure(image=img_motivation_zero)
        text_lbl.imgtk = img_motivation_zero
        interval = random.randint(200,300)
        num_frame = -1
    num_frame = num_frame + 1

    loop = screen.after(1, video_stream) 


def countdown():
    global n_countdown
    milliseconds = 5000
    if(n_countdown == 31):
        button_inizia_adesso.place_forget()
        milliseconds = 1000
    if (n_countdown > -2):
        countdown_image = get_resized_image("src/images/countdown_" + str(n_countdown) + ".png",800)
        screen.configure(image=countdown_image)
        screen.imgtk = countdown_image
        n_countdown = n_countdown - 1
        screen.after(milliseconds,countdown)
    else: 
        img = get_resized_image('src/images/bg_camera.png',800)
        screen.imgtk = img
        screen.configure(image=img)
        text_lbl.place(x = 330, y=450)
        class_lbl.place(x = 20, y = 80)
        n_countdown = 31
        date_workout = date.today()
        date_workout = date_workout.strftime("%d/%m/%Y")
        user_data.append(date_workout)
        video_stream()
    
def review_schedule(level):
    button_principiante.place_forget()
    button_intermedio.place_forget()
    button_avanzato.place_forget()
 
    bg_scheda = get_resized_image("src/images/scheda_" + level + ".png",800)
    screen.configure(image=bg_scheda)
    screen.imgtk = bg_scheda

    button_inizia_adesso.place(x=280, y=410)


def principiante():
    global tot_reps
    tot_reps = 10
    review_schedule("principiante")

def intermedio():
    global tot_reps
    tot_reps = 20
    review_schedule("intermedio")

def avanzato():
    global tot_reps
    tot_reps = 30
    review_schedule("avanzato")


def level_of_training():
    button_profilo.place_forget()
    button_vai_allenamento.place_forget()
    bg_level = get_resized_image('src/images/bg_2.png',800)
    screen.configure(image=bg_level)
    screen.imgtk = bg_level

    button_principiante.place(x=40, y=350)
    button_intermedio.place(x=290, y=350)
    button_avanzato.place(x=540, y=350)

def profilo():
    global profilo_watched
    global badges
    profilo_watched = True

    button_profilo.place_forget()
    button_vai_allenamento.place_forget()

    profilo_bg = get_resized_image('src/images/profilo_bg.png',800)
    screen.configure(image=profilo_bg)
    screen.imgtk = profilo_bg

    #visualizzare dati
    history = load_csv()
    space = "                "
    txt = ""
    for i in range(len(history[0])):
        txt = txt + history[len(history)-1][i]
        if i != len(history[0]) - 1:
            txt= txt + space
    lbl_history1.configure(text=txt)
    lbl_history1.place(x = 10, y = 80)

    txt = ""
    for i in range(len(history[0])):
        txt = txt + history[len(history)-2][i]
        if i != len(history[0]) - 1:
            txt= txt + space
    lbl_history2.configure(text=txt)
    lbl_history2.place(x = 10, y = 110)

    txt = ""
    for i in range(len(history[0])):
        txt = txt + history[len(history)-3][i]
        if i != len(history[0]) - 1:
            txt= txt + space
    lbl_history3.configure(text=txt)
    lbl_history3.place(x = 10, y = 140)

    txt = ""
    for i in range(len(history[0])):
        txt = txt + history[len(history)-4][i]
        if i != len(history[0]) - 1:
            txt= txt + space
    lbl_history4.configure(text=txt)
    lbl_history4.place(x = 10, y = 170)
    
    txt = ""
    for i in range(len(history[0])):
        txt = txt + history[len(history)-5][i]
        if i != len(history[0]) - 1:
            txt= txt + space
    lbl_history5.configure(text=txt)
    lbl_history5.place(x = 10, y = 200)

    if len(badges) == 0:
        with open(badges_path, 'r') as file:
            badges_doc = yaml.load(file, Loader=yaml.FullLoader)
            for index in range(len(badges_doc)):
                keys = list(badges_doc[index].keys())
                if (keys[0][0:2] != "n_"):
                    values = list(badges_doc[index].values())
                    badges[keys[0]] = values[0]

    img_badge1 = get_resized_image(str("src/images/"+ classes[0] 
            +"_"+ badges[classes[0]] +".png"),220)
    lbl_badge1.configure(image=img_badge1)
    lbl_badge1.imgtk = img_badge1
    lbl_badge1.place(x=20, y=320)

    img_badge2 = get_resized_image(str("src/images/"+ classes[1] 
            +"_"+ badges[classes[1]] +".png"),220)
    lbl_badge2.configure(image=img_badge2)
    lbl_badge2.imgtk = img_badge2
    lbl_badge2.place(x=295, y=320)

    img_badge3 = get_resized_image(str("src/images/"+ classes[2] 
            +"_"+ badges[classes[2]] +".png"),220)
    lbl_badge3.configure(image=img_badge3)
    lbl_badge3.imgtk = img_badge3
    lbl_badge3.place(x=560, y=320)

    img_badge4 = get_resized_image(str("src/images/"+ classes[3] 
            +"_"+ badges[classes[3]] +".png"),220)
    lbl_badge4.configure(image=img_badge4)
    lbl_badge4.imgtk = img_badge4
    lbl_badge4.place(x=20, y=375)

    img_badge5 = get_resized_image(str("src/images/"+ classes[4] 
            +"_"+ badges[classes[4]] +".png"),220)
    lbl_badge5.configure(image=img_badge5)
    lbl_badge5.imgtk = img_badge5
    lbl_badge5.place(x=295, y=375)

    img_badge6 = get_resized_image(str("src/images/"+ classes[5] 
            +"_"+ badges[classes[5]] +".png"),220)
    lbl_badge6.configure(image=img_badge6)
    lbl_badge6.imgtk = img_badge6
    lbl_badge6.place(x=560, y=375)

    button_indietro.place(x=313, y = 430)

def profile_or_workout():
    global fb
    global congratulazioni
    global badge_medal
    global user_data
    global profilo_watched

    if profilo_watched == True:
        profilo_watched = False
        lbl_badge1.place_forget()
        lbl_badge2.place_forget()
        lbl_badge3.place_forget()
        lbl_badge4.place_forget()
        lbl_badge5.place_forget()
        lbl_badge6.place_forget()
        button_indietro.place_forget()
        lbl_history1.place_forget()
        lbl_history2.place_forget()
        lbl_history3.place_forget()
        lbl_history4.place_forget()
        lbl_history5.place_forget()
    if fb == True:
        button_hard_ex.place_forget()
        button_dontlike_ex.place_forget()
        button_tired.place_forget()
        button_dont_say.place_forget()
        fb = False
    if congratulazioni == True:
        button_avanti.place_forget()
        root.after_cancel(loop)
        congratulazioni = False
    if badge_medal != "none":
        lbl_badge.place_forget()      
    
    badge_medal = "none"
    button_get_started.place_forget()
    bg = get_resized_image('src/images/pre_start_workout.png',800)
    screen.configure(image=bg)
    screen.imgtk = bg

    button_profilo.place(x=125, y=300)
    button_vai_allenamento.place(x=425, y=300)


# -- Detector, tracker, classifier
skeleton_detector = SkeletonDetector(OPENPOSE_MODEL, OPENPOSE_IMG_SIZE)

multiperson_tracker = Tracker()

multiperson_classifier = MultiPersonClassifier(SRC_MODEL_PATH, classes)

# -- Image reader and displayer
images_loader = select_images_loader(SRC_DATA_TYPE, SRC_DATA_PATH)

# interface
root = Tk()
root.geometry("800x500")
root.resizable(False, False)
# Capture from camera
cap = cv2.VideoCapture(0)

n_countdown = 31
num_frame = 0
last_motivation_number = 0
interval = random.randint(200,300)
class_idx = 0
num_reps = 0
tot_reps = 0
angle = 0
start_position = False
negative_phase = False
reached_angle = False
fb = False
congratulazioni = False
badges = {}
badge_class = -1
badge_medal = "none"
user_data = []
profilo_watched = False

img = get_resized_image('src\images\screen.png',800)
screen = Label(root, image=img, borderwidth=0)
screen.place(x=0,y=0, relwidth=1, relheight=1)

camera_lbl = Label(root, borderwidth=0)

img_motivation_zero = get_resized_image("src/images/motivation_0.png", 220)
text_lbl = Label(root, image=img_motivation_zero, borderwidth=0)

img_button = get_resized_image('src\images\continua.png',250)
button_get_started = Button(root, image=img_button, 
    bg="white",command=profile_or_workout,activebackground='white', borderwidth=4)
button_get_started.place(x=280, y=400)

img_profilo = get_resized_image('src\images\profilo.png',250)
button_profilo = Button(root, image=img_profilo, 
    bg="white",command=profilo, activebackground='white', borderwidth=4)

img_vai_allenamento = get_resized_image('src\images\inizia_allenamento.png',250)
button_vai_allenamento = Button(root, image=img_vai_allenamento, 
    bg="white",command=level_of_training, activebackground='white', borderwidth=4)

img_button_principiante = get_resized_image('src\images\principiante.png',220)
button_principiante = Button(root, image=img_button_principiante,
    bg="white",command=principiante,activebackground='white', borderwidth=4)

img_button_intermedio = get_resized_image('src\images\intermedio.png',220)
button_intermedio = Button(root, image=img_button_intermedio,
    bg="white",command=intermedio,activebackground='white', borderwidth=4)


img_button_avanzato = get_resized_image('src/images/avanzato.png',220)
button_avanzato = Button(root, image=img_button_avanzato,
    bg="white",command=avanzato,activebackground='white', borderwidth=4)

img_inizia_adesso = get_resized_image('src/images/inizia_adesso.png',220)
button_inizia_adesso = Button(root, image=img_inizia_adesso,
    bg="white",command=countdown,activebackground='white', borderwidth=4)

img_concludi = get_resized_image('src/images/concludi_allenamento.png',170)
button_concludi = Button(root, image=img_concludi,
    bg="white",command=terminate_workout,activebackground='white', borderwidth=0)

img_initial_class = get_resized_image("src/images/" + str(classes[class_idx])+ ".png", 170)
class_lbl = Label(root, image=img_initial_class, borderwidth=0)

reps_lbl = Label(root, text="", bg="white", borderwidth = 0, font=("Helvetica", 50, "bold"))


img_hard_ex = get_resized_image('src/images/hard_ex.png',220)
button_hard_ex = Button(root, image=img_hard_ex,
    bg="#343434",command=feedback,activebackground='white', borderwidth=0)

img_dontlike_ex = get_resized_image('src/images/dont_like_ex.png',220)
button_dontlike_ex = Button(root, image=img_dontlike_ex,
    bg="#343434",command=feedback,activebackground='white', borderwidth=0)

img_tired = get_resized_image('src/images/tired.png',220)
button_tired = Button(root, image=img_tired,
    bg="#343434",command=feedback,activebackground='white', borderwidth=0)

img_dont_say = get_resized_image('src/images/dont_say.png',220)
button_dont_say = Button(root, image=img_dont_say,
    bg="#343434",command=feedback,activebackground='white', borderwidth=0)

img_pop_continue = get_resized_image('src/images/pop_continue.png',220)
img_pop_stop = get_resized_image('src/images/pop_stop.png',220)

lbl_badge = Label(root, borderwidth=0)

img_avanti = get_resized_image('src/images/avanti.png',250)
button_avanti = Button(root, image=img_avanti,
    bg="white",command = profile_or_workout, activebackground='white', borderwidth=4)

lbl_badge1 = Label(root,borderwidth=0)
lbl_badge2 = Label(root,borderwidth=0)
lbl_badge3 = Label(root,borderwidth=0)
lbl_badge4 = Label(root,borderwidth=0)
lbl_badge5 = Label(root,borderwidth=0)
lbl_badge6 = Label(root,borderwidth=0)

img_indietro = get_resized_image('src/images/indietro.png',180)
button_indietro = Button(root, image=img_indietro,
    bg="white",command = profile_or_workout,activebackground='white', borderwidth=4)

lbl_history1 = Label(root, text="", bg="white", borderwidth = 0, font=("Helvetica", 14, "bold"))
lbl_history2 = Label(root, text="", bg="white", borderwidth = 0, font=("Helvetica", 14, "bold"))
lbl_history3 = Label(root, text="", bg="white", borderwidth = 0, font=("Helvetica", 14, "bold"))
lbl_history4 = Label(root, text="", bg="white", borderwidth = 0, font=("Helvetica", 14, "bold"))
lbl_history5 = Label(root, text="", bg="white", borderwidth = 0, font=("Helvetica", 14, "bold"))

root.mainloop()



