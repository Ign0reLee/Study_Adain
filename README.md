# Study_Adain
<br>
<br>Study Repository For Adaptive Instance Normalization

<br>Original Paper : [link](https://arxiv.org/abs/1703.06868)

<br>Original Code : [link](https://github.com/xunhuang1995/AdaIN-style)

<br>Data Set : [MSCOCO](https://cocodataset.org/#download), [WikiArt](https://www.kaggle.com/c/painter-by-numbers/data?select=train.zip)

<br><br>

# Enviroment
<br> 
Tensorflow 2.1.0<br> 
opencv-python >= 3.4.0<br> 
numpy >= 18.0.4<br> 
glob<br> 

<br><br>

# Usage
<br>

```bash
python train.py --ct ContentPathYouWant --st StylePathYouWant
```

optional

- cw : content weight (default: 1.0)<br>
- sw : style weight (default: 10.0)<br>
- lr : learning rate (default: 1e-4)<br>
- batch_size : batch size (default: 8)<br>
- epochs : epochs (default: 2)<br>
- visual_dir : check points path (default: ./Visualization_Training/)<br>
- save_dir : check points path (default: ./CheckPoints/)<br>

<br><br>

# Testing
<br>

```bash
python test.py --ct ContentImagePathYouWant --st StyleImagePathYouWant
```

optional

- load_index : batch size (default: -1)<br>
- load_dir : learning rate (default: ./CheckPoints/)<br>
- out_dir : epochs (default: ./Outputs/)<br>



