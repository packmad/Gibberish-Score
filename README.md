# Gibberish-Score

[https://pypi.org/project/gibberish-score/](https://pypi.org/project/gibberish-score/)
```shell
pip install gibberish-score
```

```python
from gibberish_score.gibberish_score import GibberishScore, model_builder
model = model_builder('datasets/english_words.txt')
gs: GibberishScore(model)

```