import ctypes
from ctypes.util import find_library
import os
import speech_recognition as sr
from rins.Project_RINS_2.speech_intent import get_intent

ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(
    None,
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_char_p,
)

def py_error_handler(filename, line, function, err, fmt):
    pass

c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)

asound = ctypes.cdll.LoadLibrary(find_library("asound"))
asound.snd_lib_error_set_handler(c_error_handler)

# vse zgori je sam zato da ne izpisuje napake in warninge v terminal ker je annoying



# print(sr.Microphone.list_microphone_names())

r = sr.Recognizer()




RESPONSES = {
    "BARRELS": "Inspecting the barrels now.",
    "RINGS": "Counting the rings now.",
    "ANOMALY_RED": "Detecting anomalies in the red cell now.",
    "ANOMALY_GREEN": "Detecting anomalies in the green cell now.",
    "YES": "Affirmative.",
    "NO": "Negative.",
    "NOTHING": "No task will be performed."
}

gender = "male"
start_conversation = True
pending_task = None
task_intents = {"BARRELS", "RINGS", "ANOMALY_RED", "ANOMALY_GREEN", "NOTHING"}
response_count = {
    "BARRELS": 0,
    "RINGS": 0,
    "ANOMALY_RED": 0,
    "ANOMALY_GREEN": 0,
    "NOTHING": 0,
}

def reset_response_counts():
    for key in response_count:
        response_count[key] = 0

while True:

    if start_conversation:
        if gender == "male":
            print("Hi man, which task should I perform?")
        else:
            print("Hi woman, which task should I perform?")    
        start_conversation = False

    try:
        with sr.Microphone(device_index=9) as source:
            print("Listening...")
            
            r.adjust_for_ambient_noise(source, duration=0.2)
            audio = r.listen(source)
            text = r.recognize_google(audio)
            text = text.lower()  
        
            with open("transcript.txt", "a") as f:
                f.write(text)
                f.write("\n")
            print("You said:", text)
            intent, detected_affirmation = get_intent(text)

            if gender == "female":
                if detected_affirmation is True:
                    if pending_task is not None:
                        print(RESPONSES.get(pending_task))
                        reset_response_counts()
                        pending_task = None
                elif detected_affirmation is False:
                    pending_task = None
                    reset_response_counts()
                elif intent in task_intents:
                    pending_task = intent
                    response_count[intent] += 1

                    if response_count[intent] >= 2:
                        print(RESPONSES.get(pending_task))
                        reset_response_counts()
                        pending_task = None
                    else:
                        print("Are you sure you want me to perform this task?")
            else:
                if intent in task_intents:
                    print(RESPONSES.get(intent))


    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))

    except sr.UnknownValueError:
        print("Could not understand audio")

    except KeyboardInterrupt:
        print("Program terminated by user")
        with open("transcript.txt", "a") as f:
            f.flush()
            f.close()
        if os.path.exists("transcript.txt"):
            os.remove("transcript.txt")
            print("File deleted.")
        break
    
