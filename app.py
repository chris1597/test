from flask import Flask, request, jsonify
import whisper
import soundfile as sf
from io import BytesIO

app = Flask(__name__)
model = whisper.load_model("base")

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Read the file into a buffer
    buffer = BytesIO()
    file.save(buffer)
    buffer.seek(0)

    # Load the buffer content as a numpy array
    audio, sr = sf.read(buffer, dtype='float32')

    # Transcribe using Whisper
    result = model.transcribe(audio)
    return jsonify({'transcription': result["text"]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
