# LEMMA: Laplacian pyramids for Efficient Marine SeMAntic Segmentation

## Abstract

Semantic segmentation in marine environments is crucial for the autonomous navigation of unmanned surface vessels (USVs) and detecting environmental hazards such as oil spills. However, existing methods, often relying on deep CNNs and transformer-based architectures, face challenges in deployment due to their high computational costs and resource-intensive nature. These limitations hinder the practicality of real-time, low-cost applications in real-world marine settings.

To address this, we propose LEMMA, a lightweight semantic segmentation model designed specifically for accurate environmental perception in marine imagery. The proposed architecture leverages Laplacian Pyramids to enhance edge recognition, a critical component for effective feature extraction in complex marine environments. By integrating edge information early in the feature extraction process, LEMMA eliminates the need for computationally expensive feature map computations in deeper network layers, drastically reducing model size, complexity and inference time. LEMMA demonstrates state-of-the-art performance across datasets captured from diverse platforms while reducing trainable parameters and computational requirements by up to 71x, GFLOPs by upto 77.69%, and GFLOPs by upto 86.25%, as compared to existing models. Experimental results highlight its effectiveness and real-world applicability, including 0.9342 IoU on the Oil Spill dataset and 0.9896 mIoU on Mastr1325. 

## To run the code:

Download datasets from: <br>
Mastr1325 - https://box.vicos.si/borja/viamaro/index.html <br>
Oil-Spill-Drone - https://zenodo.org/records/10555314 <br>

1. Clone the repo
2. %cd into the repo
3. Run:
```bash
python train.py --\path\to\images --\path\to\masks
```
