import wave
from piper import PiperVoice, SynthesisConfig
from pydub import AudioSegment
from pydub.playback import play

voice = PiperVoice.load("./en_US-lessac-medium.onnx")

with wave.open("spill.wav", "wb") as wav_file:
    voice.synthesize_wav("I detected a spill when inspecting the barrels.", wav_file)

sound = AudioSegment.from_file("spill.wav", format="wav")
sound = sound.set_frame_rate(48000)
play(sound)
# song = AudioSegment.from_wav("never_gonna_give_you_up.wav")


# syn_config = SynthesisConfig(
#     volume=0.5,  # half as loud
#     length_scale=2.0,  # twice as slow
#     noise_scale=1.0,  # more audio variation
#     noise_w_scale=1.0,  # more speaking variation
#     normalize_audio=False, # use raw audio from voice
# )

# voice.synthesize_wav(text="test",wav_file="test.wav", syn_config=syn_config)
