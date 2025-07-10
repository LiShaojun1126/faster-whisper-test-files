from faster_whisper import WhisperModel
from datetime import datetime
import os

model_size = "tiny.en"

model = WhisperModel(model_size, device="cpu", compute_type="int8")

segments, info = model.transcribe("/workspace/faster-whisper/faster-whisper-test-files/Sunny Yang Notification Recordings 6-30-25/Example 1.wav", beam_size=5)
segments = list(segments)
total_words = sum(len(segment.text.split()) for segment in segments)

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
print("Total words are:" + str(total_words))

timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S") #use UTC time zone
filename = f"log_{timestamp}.txt"

os.makedirs("logs", exist_ok=True)
filepath = os.path.join("logs", filename)

with open(filepath, "a", encoding="utf-8") as f:
    for segment in segments:
        f.write(segment.text + "\n")
    f.write("Total words:" + str(total_words))
