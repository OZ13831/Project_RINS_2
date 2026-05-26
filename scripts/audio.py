import wave
from piper import PiperVoice
from pydub import AudioSegment


voice = PiperVoice.load("./en_US-lessac-medium.onnx")


def synth(text: str, filename: str) -> None:
    with wave.open(filename, "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)
    sound = AudioSegment.from_file(filename, format="wav")
    sound = sound.set_frame_rate(48000)
    sound.export(filename, format="wav")


# ── Digits 0–9 ───────────────────────────────────────────────────────────────
synth("zero",  "0.wav")
synth("one",   "1.wav")
synth("two",   "2.wav")
synth("three", "3.wav")
synth("four",  "4.wav")
synth("five",  "5.wav")
synth("six",   "6.wav")
synth("seven", "7.wav")
synth("eight", "8.wav")
synth("nine",  "9.wav")

# ── Fixed phrases ─────────────────────────────────────────────────────────────
synth("Are you sure you want me to perform this task?", "confirm_task.wav")
synth("Hi woman, which task should I perform?",         "hi_woman.wav")
synth("Hi man, which task should I perform?",           "hi_man.wav")
synth("Warning, spilt barrel.",                         "warning_spilt_barrel.wav")
synth("Here is the report.",                            "here_is_the_report.wav")

# ── Split fragments for phrases with numeric placeholders ─────────────────────
# "There were [x] rings"   → there_were.wav + N.wav + rings.wav
# "There were [x] barrels" → there_were.wav + N.wav + barrels.wav
synth("There were", "there_were.wav")
synth("rings.",     "rings.wav")
synth("barrels.",   "barrels.wav")

# "There were [x] anomalies in [y] tiles"
# → there_were.wav + N.wav + anomalies_in.wav + M.wav + tiles.wav
synth("anomalies in", "anomalies_in.wav")
synth("tiles.",       "tiles.wav")

# ── Existing spill phrase ─────────────────────────────────────────────────────
synth("I detected a spill when inspecting the barrels.", "spill.wav")

print("All audio files generated.")
