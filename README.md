<div align="center">

# MIQD-2.5M

[![Dataset](https://img.shields.io/badge/Dataset-Available-brightgreen?logo=github)](https://github.com/XiaoqiWang/MIQD-2.5M)
[![Code](https://img.shields.io/badge/Code-Available-orange?logo=github)](https://github.com/XiaoqiWang/MIQA)
[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b?logo=arxiv)](https://arxiv.org/abs/2508.19850)
[![Model](https://img.shields.io/badge/Model-HuggingFace-yellow?logo=huggingface)](https://huggingface.co/)
[![Stars](https://img.shields.io/github/stars/XiaoqiWang/MIQD-2.5M?style=social&logo=github)](https://github.com/XiaoqiWang/MIQD-2.5M)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?logo=open-source-initiative)](LICENSE)

</div>

---

<div align="center">


<p align="center" style="color:gray; margin-bottom:0.01em; font-size:1.2em">
This repository provides the database for the paper:
</p>
<h2 align="center" style="margin-top:0.01em;">
Image Quality Assessment for Machines: Paradigm, Large-scale Database, and Models
</h2>

*Xiaoqi Wang, Yun Zhang, Weisi Lin*  

[📖 Paper](https://arxiv.org/abs/2508.19850) | [🗃️ Dataset](https://github.com/XiaoqiWang/MIQD-2.5M) | [👨‍💻 Code](https://github.com/XiaoqiWang/MIQA) | [🤗 HuggingFace](https://huggingface.co/)  

</div>


---

## 📋 Database Construction Pipeline
<div align="center">
  <img src="./assets/database.bmp" alt="Example Image" width="1400"/>
</div>

## ✨ Database Highlights

- **Massive scale**: 2.5 million degraded images generated from 10,000 carefully selected original images across 1,000+ object categories.
- **Region-aware degradation**: Introduces spatial degradation patterns (uniform, ROI-dominated, and background-dominated) to simulate real-world quality variations.
- **Comprehensive degradation**: 10 distortion types across 4 categories (digital, blur, environmental, noise) with 5 severity levels each.
- **Multi-task evaluation**: Covering three representative vision tasks (image classification, object detection, and instance segmentation) with a total of 75 diverse models.
- **Quality labels**: Machine vision quality is evaluated by consistency (prediction stability), accuracy (task performance), and their integrated composite metric.
---

### 📝 Dataset Description 
<details open>
<summary>📊 MIQD-2.5M Characteristics Summary</summary>

<sub> 

| Task | Original Images | Source | Degradation Types | Severity Levels | Region Types | Object Categories | Degraded Images | Resolution Range |
|------|-----------------|--------|-------------------|-----------------|--------------|-------------------|-----------------|------------------|
| Image Classification | 5,000 | ImageNet | 10 | 5 | 3 | 1,000 | 1,250,000 | 262×415 ~ 4288×2848 |
| Object Detection & Instance Segmentation | 5,000 | MS COCO | 10 | 5 | 3 | 80 | 1,250,000 | 200×145 ~ 640×640 |
| **Total** | **10,000** | **-** | **10** | **5** | **3** | **≥1,000** | **2,500,000** | **200×145 ~ 4288×2848** |
</sub>
</details> 

<details >
<summary>📊 Summary of Degradation Regions, Distortion Intensity Levels, and Categories  </summary>

<sub>
Each pair **(x, y)** represents ROI distortion intensity `x` and background distortion intensity `y`.
<table>
  <thead>
    <tr>
      <th>Degraded Region</th>
      <th>Degraded Severity</th>
      <th>Degraded Region</th>
      <th>Degraded Severity</th>
      <th>Degraded Region</th>
      <th>Degraded Severity</th>
      <th>Degraded Category</th>
      <th>Degraded Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="10"><b>Uniform Distortion (UD)</b></td>
      <td>(1, 1)</td>
      <td rowspan="10"><b>ROI-Dominated Distortion (ROI-DD)</b></td>
      <td>(2, 1)</td>
      <td rowspan="10"><b>Background-Dominated Distortion (BG-DD)</b></td>
      <td>(1, 2)</td>
      <td rowspan="3">Digital</td>
      <td>Contrast</td>
    </tr>
    <tr>
      <td></td>
      <td>(3, 1)</td>
      <td>(1, 3)</td>
      <td>Pixelate</td>
    </tr>
    <tr>
      <td>(2, 2)</td>
      <td>(4, 1)</td>
      <td>(1, 4)</td>
      <td>JPEG</td>
    </tr>
    <tr>
      <td></td>
      <td>(5, 1)</td>
      <td>(1, 5)</td>
      <td rowspan="3">Blur</td>
      <td>Motion blur</td>
    </tr>
    <tr>
      <td>(3, 3)</td>
      <td>(3, 2)</td>
      <td>(2, 3)</td>
      <td>Defocus blur</td>
    </tr>
    <tr>
      <td></td>
      <td>(4, 2)</td>
      <td>(2, 4)</td>
      <td>Glass blur</td>
    </tr>
    <tr>
      <td>(4, 4)</td>
      <td>(5, 2)</td>
      <td>(2, 5)</td>
      <td rowspan="3">Environmental Conditions</td>
      <td>Fog</td>
    </tr>
    <tr>
      <td></td>
      <td>(4, 3)</td>
      <td>(3, 4)</td>
      <td>Snow</td>
    </tr>
    <tr>
      <td>(5, 5)</td>
      <td>(5, 3)</td>
      <td>(3, 5)</td>
      <td>Darkness</td>
    </tr>
    <tr>
      <td></td>
      <td>(5, 4)</td>
      <td>(4, 5)</td>
      <td>Noise</td>
      <td>Gaussian noise</td>
    </tr>
  </tbody>
</table>
</sub>
</details> 

---

## 📥 Download
**The full dataset (MIQD-2.5M) will be released publicly in the near future.** Please stay tuned for updates.  
### 🔗 Download Links

|    **Dataset**    |       **Task**        |              **Images**               | **Labels** | **Additional Info** | **Original Images** | **Full Dataset** | **Size** |
|:-----------------:|:---------------------:|:-------------------------------------:|:----------:|:-------------------:|:-------------------:|:----------------:|:--------:|
| MIQD-2.5M Subset1 | image classification  | [Google Drive](链接) / [Baidu Pan](链接) | [Google Drive](链接) / [Baidu Pan](链接) | [Google Drive](链接) / [Baidu Pan](链接) | [Google Drive](链接) / [Baidu Pan](链接) | - | ~800 GB |
| MIQD-2.5M Subset2 |   object detection    | [Google Drive](链接) / [Baidu Pan](链接) | [Google Drive](链接) / [Baidu Pan](链接) | [Google Drive](链接) / [Baidu Pan](链接) | [Google Drive](链接) / [Baidu Pan](链接) | - | ~600 GB |
| MIQD-2.5M Subset3 | instance segmentation | [Google Drive](链接) / [Baidu Pan](链接) | [Google Drive](链接) / [Baidu Pan](链接) | [Google Drive](链接) / [Baidu Pan](链接) | [Google Drive](链接) / [Baidu Pan](链接) | - | ~700 GB |
| <span style="color:red">**MIQD-2.5M (Full)**</span> | <span style="color:red">**All Tasks**</span> | - | - | - | [Google Drive](链接) / [Baidu Pan](链接) | [Google Drive](全集链接) / [Baidu Pan](全集链接) | **~2.1 TB** |
**📌 Notes:**  
You can **download either individual subsets** (e.g., classification, detection, segmentation) or **the complete dataset**. We provide multiple hosting options (**Google Drive** and **Baidu Pan**) to ensure stable access.

### 📂 Dataset Structure
<details>
<summary>Dataset Structure</summary>

<sub> 

```
MIQD_2.5M/
├── miqd_cls/                        # Image classification subset
│   ├── images/                      
│   │   ├── folder1/
│   │   ├── folder2/
│   │   └── ...
│   ├── labels/                     
│   │   └── cls_labels.csv
│   └── metadata/                    
│       └── cls_info.json
│
├── miqd_det/                        # Object detection subset
.......
│
├── miqd_ins/                        # Instance segmentation subset
 ......
│
└── README.md

```
</sub>
</details>

### 🧪 Dataset Samples
<details>
<summary>Dataset Samples</summary>

<sub> 
<div align="center">
  <img src="./assets/examples.png" alt="Example Image" width="1400"/>
</div>
Sample illustration showcasing the corresponding  
<span style="color:#9ACD32; font-weight:bold">PSNR</span>,  
<span style="color:#556B2F; font-weight:bold">SSIM</span>, and  
<span style="color:#006400; font-weight:bold">LPIPS</span> values,
along with the predicted <span style="color:#6495ED; font-weight:bold"> Consistency</span>,  
<span style="color:#000080; font-weight:bold">Accuracy</span>, and <span style="color:#191970; font-weight:bold">Composite</span> scores, and their respective ground-truth<span style="color:#FF7F50; font-weight:bold"> Consistency</span>, <span style="color:#FF4500; font-weight:bold">Accuracy</span>, and<span style="color:#8B4513; font-weight:bold"> Composite</span> scores.  
Panels (a)–(d), (e)–(g), and (h)–(j) present example images along with MIQA-related scores for image classification under UD, object detection under ROI-DD, and instance segmentation under BG-DD, respectively. Numbers in parentheses indicate distortion severity in ROI and background regions.
Note: Lower LPIPS indicates higher perceptual quality, whereas higher values are preferred for other metrics.

</sub>
</details>

---

## 📚 Citation

If you find our work useful in your research, please consider citing:

```bibtex
@article{wang2025miqa,
  title={Image Quality Assessment for Machines: Paradigm, Large-scale Database, and Models},
  author={Wang, Xiaoqi and Zhang, Yun and Lin, Weisi},
  journal={arXiv preprint arXiv:2508.19850},
  year={2025}
}
```

---

## 📮 Contact

For questions and feedback:

- **📧 Email**: wangxq79@mail2.sysu.edu.cn; xqwang.research@outlook.com

---

## 📄 License and Terms of Use

<details>
<summary>License and Terms of Use</summary>

<sub>

- License: This project is released under the [MIT License]. See the LICENSE file for details.
- Academic Use Only: The dataset is released solely for non-commercial research and educational purposes.
- Data Sources Disclaimer: All source images originate from publicly available datasets, and copyright remains with the original owners.
- Citation Requirement: Users must cite the associated paper(s) when using the dataset in publications.
- Responsibility: Users are fully responsible for compliance with local laws and ethical standards when using the dataset.

</sub>
</details>
 
---

<div align="center">
  <sub>Built with ❤️ by [Xiaoqi Wang]</sub>
</div>