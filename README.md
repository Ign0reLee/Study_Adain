# Study_Adain
---

<br>Study Repository For Adaptive Instance Normalization

<br>Original Paper : [link](https://arxiv.org/abs/1703.06868)

<br>Original Code : [link](https://github.com/xunhuang1995/AdaIN-style)

<br>Data Set : [MSCOCO](https://cocodataset.org/#download), [WikiArt](https://www.kaggle.com/c/painter-by-numbers/data?select=train.zip)

<br><br>
---
<br><br>

# Enviroment

Tensorflow 2.1.0

opencv-python >= 3.4.0

numpy >= 18.0.4

glob

<br><br>
---
<br><br>

# Usage

```python
train.py --ct ContentPathYouWant --st StylePathYouWant
```

optional

- lr : learning rate (default: 1e-4)<br>
- epochs : epochs (default: 2)<br>
- cw : content weight (default: 1.0)<br>
- sw : style weight (default: 10.0)<br>
- save_dir : check points path (default: ./CheckPoints/)<br>
- bathc_size : batch size (default: 8)<br>


