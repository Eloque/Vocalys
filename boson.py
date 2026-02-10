from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
from boson_multimodal.data_types import ChatMLSample, Message
import torchaudio
import torch

# Model and tokenizer names
MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cuda"

# Prepare a small “chat” sample with narration-style prompt
messages = [
    Message(role="system", content="Generate audio following instruction."),
    Message(role="user", content="The frost clung to the stone walls of Frosthaven. It was cold and damp, mercenaries running around franticly.ex"),
]



serve_engine = HiggsAudioServeEngine(MODEL_PATH, TOKENIZER_PATH, device=device)

output = serve_engine.generate(
    chat_ml_sample=ChatMLSample(messages=messages),
    max_new_tokens=1024,
    temperature=0.3,
    top_p=0.95,
    top_k=50,
)

# Save audio
audio_tensor = torch.from_numpy(output.audio)[None, :]
torchaudio.save("higgs_output.wav", audio_tensor, output.sampling_rate)

print("Audio saved as higgs_output.wav")
