# AI-PROJECT (Hopefully Last)

## How to Import belief_func.py
First Import
```python
from agents.belief_node import BeliefNode
```

Then Initialize it once
```python
belief_node = BeliefNode(
    num_classes=6,
    alpha=0.6,
    activity_names=[
        "Walking",
        "Jogging",
        "Upstairs",
        "Downstairs",
        "Sitting",
        "Standing"
    ]
)
```

After This Get the Model Output
```python
raw_probs = model.predict([X, C])[0] # From model_predict.py
```

Get Smoothed Out Probability
```python
smoothed_probs = belief_node.update(raw_probs)
```

Get the Final Activity
```python
idx, confidence = belief_node.top
final_activity = belief_node.activity_names[idx]
```