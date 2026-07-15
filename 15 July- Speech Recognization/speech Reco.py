import pyttsx3 as pt
import pywhatkit as pk

engine = pt.init()
engine.setProperty("rate", 170)

def speak(text):
    engine.say(text)
    engine.runAndWait()

def hear():
    cmd = input("Enter your command: ").lower()
    return cmd

def run():
    cmd = hear()

    if "play" in cmd:
        song = cmd.replace("play", "").strip()
        speak(f"Playing {song}")
        pk.playonyt(song)
    else:
        speak("Sorry, I don't know that command.")

run()