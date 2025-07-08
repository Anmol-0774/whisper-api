import os
import whisper
from flask import Flask, request, send_file
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = whisper.load_model("small")

@app.route("/upload", methods=["POST"])
def upload_audio():
    if 'audio' not in request.files:
        return {"error": "No audio file part"}, 400

    file = request.files['audio']
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    result = model.transcribe(filepath, language="ur", task="transcribe")
    srt_path = filepath.replace(".mp4", ".srt").replace(".wav", ".srt")

    with open(srt_path, "w", encoding="utf-8") as f:
        for segment in result["segments"]:
            f.write(f"{segment['id'] + 1}\n")
            f.write(f"{format_time(segment['start'])} --> {format_time(segment['end'])}\n")
            f.write(segment['text'].strip() + "\n\n")

    return send_file(srt_path, as_attachment=True)

def format_time(seconds: float) -> str:
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hrs:02}:{mins:02}:{secs:02},{millis:03}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
