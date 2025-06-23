# Install Whisper and dependencies
!pip install git+https://github.com/openai/whisper.git 
!sudo apt update && sudo apt install ffmpeg -y

# Import libraries
import whisper
import os
from IPython.display import Audio

# Load the model (use "base", "small", "medium", or "large")
model = whisper.load_model("base")

# Upload an audio file
from google.colab import files
uploaded = files.upload()

# Get the filename
filename = next(iter(uploaded))

# Play audio in notebook (optional)
display(Audio(filename))

# Transcribe audio
result = model.transcribe(filename)
print("📝 Transcription:")
print(result["text"])
