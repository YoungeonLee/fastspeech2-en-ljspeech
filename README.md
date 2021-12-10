## Example to download fastspeech2 from fairseq

Weights are downloaded from:

We still need to git clone this repo first before being able to download it.
Having `cd`'ed into the repo we can do the following:

```python
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf

model = load_model_ensemble_and_task_from_hf("patrickvonplaten/fairseq-fastspeech2")
```