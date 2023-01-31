import pyttsx3 as a
import datetime
import speech_recognition as sr
import wikipedia
import webbrowser
import os
import cv2
import face_recognition
import numpy as np
from deepface import DeepFace
import nltk #natural language processing library 
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import requests
from bs4 import BeautifulSoup

from flask import Flask, render_template, Response
app = Flask(__name__)


num_dicct={'zero': 0 , 'one':1, 'two': 2 , 'three': 3, 'four': 4, 'five':5 , 'six':6,'seven':7,'eight':8,'nine':9}
result = ''

real_estate ='https://www.ibef.org/industry/real-estate-india.aspx'
stock="https://www.indiainfoline.com/article/news-top-story/markets-rally-as-rbi-policy-q2-earnings-take-focus-sensex-tops-390-pts-nifty-50-near-17-640-bajaj-twins-icici-bharti-airtel-dr-reddy-s-lift-121100400016_1.html"
cotton = "https://www.business-standard.com/article/economy-policy/india-s-cotton-production-to-fall-by-1-due-to-lackluster-rainfall-fitch-121091400560_1.html"
camera = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def gen_frames():
    global name , encode 
    image = face_recognition.load_image_file("tony stark.jpg")
    face_encoding = face_recognition.face_encodings(image)[0]

    known_face_encodings = [
        face_encoding
    ]
    known_face_names = [
    "Lubu"
    ]
    while True:
        try:
            success, frame = camera.read()  
        except Exception as e:
            success = False
            print(e)
        if not success:
            break
        else:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
            
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                                        
                # print(name)
                face_names.append(name)
            

            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            # print(frame)
            # encode = face_recognition.face_encodings(frame,face_recognition.face_locations(frame))
            # cv2.imshow("frame" , frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n' 
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
def emotion():
        while True: 
            ret, frame = camera.read()
            result = DeepFace.analyze(frame , actions =['emotion'],enforce_detection=False)
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray,1.1,4)
            for(x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            font=cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame,result['dominant_emotion'],(50,50),font,3,(0,0,255),2,cv2.LINE_4)
            # cv2.imshow('video', frame)
            # k = cv2.waitKey(30) & 0xff
            # if k == 27: # press 'ESC' to quit
            #     break
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n' 
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


            if(result['dominant_emotion']=='happy'):
                # playsound.playsound('C:/Users/Dell/Downloads/hap.mp3')
                webbrowser.open("https://www.youtube.com/watch?v=A-sfd1J8yX4")
                break
            elif(result['dominant_emotion']=='sad'):
                webbrowser.open("https://www.youtube.com/watch?v=i_k3K772Zyk")
                break
            elif(result['dominant_emotion']=='angry'):
                webbrowser.open("https://www.youtube.com/watch?v=Ux-BoW8h6BA")
                break
            elif(result['dominant_emotion']=='energetic'):
                webbrowser.open("https://www.youtube.com/watch?v=n1oaPb_UTxs")
                break
            elif(result['dominant_emotion']=='neutral'):
                webbrowser.open("https://youtu.be/9fNbjMXXYV4")
                break
            else:
                print("No songs found")




@app.route("/",methods = ['GET' , 'POST'])
def home():
    return render_template('index.html')

@app.route("/voi",methods = ['GET' , 'POST'])
def voi():
    hour = int(datetime.datetime.now().hour)
    engine = a.init()
    if hour >= 0 and hour < 12:
        engine.say("good morning sir")
        engine.runAndWait()
    elif hour >= 12 and hour < 18:
        engine.say("good afternoon sir")
        engine.runAndWait()
    else:
        engine.say("good evening sir")
        engine.runAndWait()
    engine.say("hello i am albus , how may i help you?")
    engine.runAndWait()

    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("listening...")
        r.pause_threshold = 1
        audio = r.listen(source)
        try:
            print("recognizing...")
            query = r.recognize_google(audio)
            print(query)
            while True:
                query = query.lower()
                
                if 'wikipedia' in query:
                    engine.say('Searching wikipedia')
                    engine.runAndWait()
                    # speak()
                    query = query.replace("wikipedia", "")
                    results = wikipedia.summary(query , sentences=1)
                    # speak()
                    engine.say("according to wikipedia")
                    engine.runAndWait()
                    print(results)
                    engine.say("results")
                    engine.runAndWait()
                    break
                    # speak(results)
                elif 'open youtube' in query:
                    # speak("sure sir")
                    engine.say("sure sir")
                    engine.runAndWait()
                    webbrowser.open("youtube.com")
                    break

                elif 'open google' in query:
                    # speak("sure sir")
                    engine.say("sure sir")
                    engine.runAndWait()
                    webbrowser.open("google.com")
                    break

                elif 'open stackoverflow' in query:
                    # speak("sure sir")
                    engine.say("sure sir")
                    engine.runAndWait() 
                    webbrowser.open("stackoverflow.com")
                    break

                elif 'play music' in query:
                    # speak("sure sir")
                    engine.say("sure sir")
                    engine.runAndWait()
                    music_dir = 'D:\\songs'
                    songs = os.listdir(music_dir)
                    print(songs)
                    os.startfile(os.path.join(music_dir, songs[0]))
                    break

                elif 'the time' in query:
                    strTime = datetime.datetime.now().strftime("%H:%M:%S")
                    # speak(f"Sir the time is {strTime}")
                    engine.say(f"Sir the time is {strTime}")
                    engine.runAndWait()
                    break
                
                elif 'stock market' in query:
                    all_text = ''
                    ans = {}
                    pagew = requests.get(stock)
                    soupw = BeautifulSoup(pagew.content,"html.parser")
                    for dataw in soupw.find_all("h1"):
                        all_text += dataw.get_text().strip()
                    sid = SentimentIntensityAnalyzer()
                    stop_words = set(stopwords.words('english'))
                    word_tokens = word_tokenize(all_text)
        # filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
                    filtered_sentence = []
                    for w in word_tokens:
                        if w not in stop_words:
                            filtered_sentence.append(w)
                    s = ' '.join(filtered_sentence)
                    print(s)
                    ans = sid.polarity_scores(all_text)
                    an = ans['compound']
                    if an >0.0:
                        engine.say('Market is Growing, Start Selling Soon')
                        engine.runAndWait()
                        break
                    elif an<0:
                        engine.say("Marking is falling, Start Buying")
                        engine.runAndWait()
                        break
                    elif an == 0:
                        engine.say("Market is neutral, Keep Hold")
                        engine.runAndWait()
                        break
                elif 'cotton market' in query:
                    all_text = ''
                    ans = {}
                    pagew = requests.get(cotton)
                    soupw = BeautifulSoup(pagew.content,"html.parser")
                    for dataw in soupw.find_all("h1"):
                        all_text += dataw.get_text().strip()
                    sid = SentimentIntensityAnalyzer()
                    stop_words = set(stopwords.words('english'))
                    word_tokens = word_tokenize(all_text)
        # filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
                    filtered_sentence = []
                    for w in word_tokens:
                        if w not in stop_words:
                            filtered_sentence.append(w)
                    s = ' '.join(filtered_sentence)
                    print(s)
                    ans = sid.polarity_scores(all_text)
                    an = ans['compound']
                    if an >0.0:
                        engine.say('Market is Growing, Start Selling Soon')
                        engine.runAndWait()
                        break
                    elif an<0:
                        engine.say("Marking is falling, Start Buying")
                        engine.runAndWait()
                        break
                    elif an == 0:
                        engine.say("Market is neutral, Keep Hold")
                        engine.runAndWait()
                        break
                elif 'real estate' in query:
                    all_text = ''
                    ans = {}
                    pagew = requests.get(real_estate)
                    soupw = BeautifulSoup(pagew.content,"html.parser")
                    for dataw in soupw.find_all("h1"):
                        all_text += dataw.get_text().strip()
                    sid = SentimentIntensityAnalyzer()
                    stop_words = set(stopwords.words('english'))
                    word_tokens = word_tokenize(all_text)
        # filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
                    filtered_sentence = []
                    for w in word_tokens:
                        if w not in stop_words:
                            filtered_sentence.append(w)
                    s = ' '.join(filtered_sentence)
                    print(s)
                    ans = sid.polarity_scores(all_text)
                    an = ans['compound']
                    if an >0.0:
                        engine.say('Market is Growing, Start Selling Soon')
                        engine.runAndWait()
                        break
                    elif an<0:
                        engine.say("Marking is falling, Start Buying")
                        engine.runAndWait()
                        break
                    elif an == 0:
                        engine.say("Market is neutral, Keep Hold")
                        engine.runAndWait()
                        break
                
# Calculator
                elif "plus" in query:
                    str1=query.split('')
                    val1 = num_dicct.get(str1[0])
                    val2 = num_dicct.get(str1[2])
                    result1 = val1 + val2
                    result = str(result1)
                    engine.say(result)
                    engine.runAndWait()
                    break
                
                elif "divide" in query:
                    str1=query.split('')
                    val1 = num_dicct.get(str1[0])
                    val2 = num_dicct.get(str1[2])
                    result1 = val1 / val2
                    result = str(result1)
                    engine.say(result)
                    engine.runAndWait()
                    break

                elif "subtract" in query:
                    str1=query.split('')
                    val1 = num_dicct.get(str1[0])
                    val2 = num_dicct.get(str1[2])
                    result1 = val1 - val2
                    result = str(result1)
                    engine.say(result)
                    engine.runAndWait()
                    break

                elif "multiply" in query:
                    str1=query.split('')
                    val1 = num_dicct.get(str1[0])
                    val2 = num_dicct.get(str1[2])
                    result1 = val1 * val2
                    result = str(result1)
                    engine.say(result)
                    engine.runAndWait()
                    break
                

        except Exception as e:
            print(e)
            print("say that again please")
            engine.say("say that again please")
            engine.runAndWait()
            return "None"

                
        # return render_template('speech.html', engine=engine)

@app.route('/emo')
def emo():
    return Response(emotion(), mimetype='multipart/x-mixed-replace; boundary=frame')
    # video_capture = cv2.VideoCapture(0)




@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)

    
