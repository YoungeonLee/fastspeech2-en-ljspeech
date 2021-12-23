---
framework: fairseq
task: text-to-speech
tags:
- fairseq
- audio
- text-to-speech
language: en
datasets:
- ljspeech
---
## Example to download fastspeech2 from fairseq

The following should work with fairseq's most up-to-date version in a google colab:

```python
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
import IPython.display as ipd
import torch

model_ensemble, cfg, task = load_model_ensemble_and_task_from_hf_hub(
    "facebook/fastspeech2-en-ljspeech", arg_overrides={"vocoder": "griffin_lim", "fp16": False}
)

def tokenize(text):
  import g2p_en
  tokenized = g2p_en.G2p()(text)
  tokenized = [{",": "sp", ";": "sp"}.get(p, p) for p in tokenized]
  return " ".join(p for p in tokenized if p.isalnum())
  
text = "This is a cool demo for speech synthesis, don't you think so?"

tokenized = tokenize(text)
sample = {
    "net_input": {
        "src_tokens": task.src_dict.encode_line(tokenized).view(1, -1),
        "src_lengths": torch.Tensor([len(tokenized.split())]).long(),
        "prev_output_tokens": None
        },
    "target_lengths": None,
    "speaker": None,
}
generator = task.build_generator(model_ensemble, cfg)
generation = generator.generate(model_ensemble[0], sample)
waveform = generation[0]["waveform"]

ipd.Audio(waveform, rate=task.sr)
```

See: https://colab.research.google.com/drive/1gvq4Y1urrg9QrQ9031sZIP93LKspIh_X?usp=sharing