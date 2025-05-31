<div align="center">
  <!-- <h1><b> OR-VSKC </b></h1> -->
  <!-- <h2><b> OR-VSKC </b></h2> -->
  <h2><b> Visual-Semantic Knowledge Conflicts in Operating Rooms: A Generative Dataset for Multimodal LLMs </b></h2>
</div>

<div align="center">

![](https://img.shields.io/github/last-commit/zgg2577/VS-KC?color=green)
![](https://img.shields.io/github/stars/zgg2577/VS-KC?color=yellow)
![](https://img.shields.io/github/forks/zgg2577/VS-KC?color=lightblue)

</div>

<p align="center">

<img src="./fig/main_framework.png" >

</p>

> 🌟 Please let us know if you find out a mistake or have any suggestions!

---


  
## 🗂 OR-VSKC Dataset Download
[![Download OR-VSKC Dataset](https://img.shields.io/badge/Download-OR--VSKC_Dataset-007ec6?style=for-the-badge&logo=google-drive&logoColor=white)](https://drive.google.com/uc?export=download&id=1i-u4gnDPH-Llx9-7eayfDvtl1I4Emx67)



---

## 📑 Appendix: Extended Documentation

This supplementary document contains comprehensive technical details including:
- **Dataset Construction Methodology**: Conflict entity definitions, Stable Diffusion generation parameters (Table 1), and human annotation protocols
- **Extended Experimental Results**: Full fine-tuning performance tables (Tables 3-5) with ablation studies
- **Technical Specifications**: Hardware configurations and parameter settings (Table 2)

[![Download Appendix](https://img.shields.io/badge/Technical_Appendix-PDF-DC143C?style=for-the-badge&logo=adobe-acrobat-reader&logoColor=white)](https://drive.google.com/uc?export=download&id=1Qi2t-MEZWHmMIjIdNDJE3uym65ppzWDi)

---

## ⚡ What are Visual-Semantic Knowledge Conflicts (VS-KC)?
Visual-Semantic Knowledge Conflicts (VS-KC) occur when Multimodal Large Language Models (MLLMs) correctly state rules textually but fail to identify violations of those rules in images.



<img src="./fig/case.png" width="600" >

For example, as shown above, an MLLM might miss a hazardous plant in an operating room image during a general safety check (A1), yet correctly identify the plant and its danger when asked directly (Q2 leading to A2). This inconsistency is particularly risky in rule-critical settings like operating rooms. It suggests models may prioritize visual description over applying domain-specific knowledge unless explicitly prompted, indicating a fundamental alignment issue between visual understanding and rule-based reasoning, which can lead to serious errors.

---
## Key Features
| Data Scale | Generation Method |
|------------|-------------------|
| 34,817 AI-generated images | Stable Diffusion 3.5 |
| 214 human-annotated images | Manually screened and synthesized |

## Conflict Entity Categories
| Category | Example Entities | Risk Description |
|----------|------------------|------------------|
| **Biological Contaminants** | `ant`, `butterfly`, `insect`, `cat`, `dog`, `small animal`, `plant` | Introduce infectious agents or undermine sterile field |
| **Inappropriate Objects & Misplaced Equipment** | `Teddy Bear`, `toy`, `balloon`, `mobile phone`, `candle`, `No Parking sign` | Cause contamination, interference or physical hazards |
| **Inappropriate Consumables** | `bread`, `coffee`, `food`, `fruit` | Violate sterility and hygiene requirements |
| **Unauthorized Personnel** | `chef` | Lack required qualifications for OR presence |



