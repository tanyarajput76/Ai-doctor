# if you dont use pipenv uncomment the following:
from dotenv import load_dotenv
load_dotenv()

#VoiceBot UI with Gradio
import os
import sys

# Prevent Python from importing this file (gradio.py) when 'import gradio' is executed
_script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
_to_restore = [p for p in sys.path if p == '' or os.path.abspath(p) == _script_dir]
for p in _to_restore:
    while p in sys.path:
        sys.path.remove(p)

import gradio as gr

for p in reversed(_to_restore):
    sys.path.insert(0, p)

from brain import encode_image, analyze_image_with_query
from voice import record_audio, transcribe_with_groq
from doctor_voice import text_to_speech_with_gtts, text_to_speech_with_elevenlabs

load_dotenv()

system_prompt="""You have to act as a professional doctor, i know you are not but this is for learning purpose. 
            What's in this image?. Do you find anything wrong with it medically? 
            If you make a differential, suggest some remedies for them. Donot add any numbers or special characters in 
            your response. Your response should be in one long paragraph. Also always answer as if you are answering to a real person.
            Donot say 'In the image I see' but say 'With what I see, I think you have ....'
            Dont respond as an AI model in markdown, your answer should mimic that of an actual doctor not an AI bot, 
            Keep your answer concise (max 2 sentences). No preamble, start your answer right away please"""


def process_inputs(audio_filepath, image_filepath):
    speech_to_text_output = ""
    if audio_filepath:
        try:
            speech_to_text_output = transcribe_with_groq(GROQ_API_KEY=os.environ.get("GROQ_API_KEY"), 
                                                         audio_filepath=audio_filepath,
                                                         stt_model="whisper-large-v3")
        except Exception as e:
            print(f"Transcription error: {e}")
            speech_to_text_output = f"Audio transcription failed: {e}"

    # Handle the image input
    if image_filepath:
        query_text = (system_prompt + " " + speech_to_text_output).strip()
        doctor_response = analyze_image_with_query(query=query_text, encoded_image=encode_image(image_filepath), model="qwen/qwen3.8-27b")
    else:
        doctor_response = "No image provided for me to analyze"

    voice_of_doctor = text_to_speech_with_elevenlabs(input_text=doctor_response, output_filepath="final.mp3") 

    return speech_to_text_output, doctor_response, voice_of_doctor


# Create the interface
iface = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Audio(sources=["microphone", "upload"], type="filepath"),
        gr.Image(type="filepath")
    ],
    outputs=[
        gr.Textbox(label="Speech to Text"),
        gr.Textbox(label="Doctor's Response"),
        gr.Audio(label="Doctor's Voice")
    ],
    title="AI Doctor with Vision and Voice"
)

iface.launch(debug=True)

#http://127.0.0.1:7860
