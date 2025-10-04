import sys
import pyttsx3

if __name__ == "__main__":
    text = sys.argv[1] if len(sys.argv) > 1 else ""
    if text:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
