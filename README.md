<div align="center">
  <!-- <h1><b> OR-VSKC </b></h1> -->
  <!-- <h2><b> OR-VSKC </b></h2> -->
  <h2><b> Resolving Visual-Semantic Knowledge Conflicts in Operating Rooms via Synthetic Data Guided Alignment for Surgical Risk Perception in Multimodal Large Language Models </b></h2>
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

## ⚡ What are Visual-Semantic Knowledge Conflicts (VS-KC)?
Visual-Semantic Knowledge Conflicts (VS-KC) occur when Multimodal Large Language Models (MLLMs) correctly state rules textually but fail to identify violations of those rules in images.



<img src="./fig/case.png" width="600" >

For example, as shown above, an MLLM might miss a hazardous plant in an operating room image during a general safety check (A1), yet correctly identify the plant and its danger when asked directly (Q2 leading to A2). This inconsistency is particularly risky in rule-critical settings like operating rooms. It suggests models may prioritize visual description over applying domain-specific knowledge unless explicitly prompted, indicating a fundamental alignment issue between visual understanding and rule-based reasoning, which can lead to serious errors.

---
# 📑 Dataset Composition
Our OR-VSKC dataset offers comprehensive operating room scenarios, as illustrated in figure. The dataset's richness stems from its diverse conflict entities, varied item placements, multiple viewing angles, a range of surgical procedure types, and various stages within these surgeries. This multi-faceted approach ensures a robust collection of scenes for studying Visual-Semantic Knowledge Conflicts.

<img src="./fig/examples.jpg" width="600" >

<img src="./fig/dataset.png" width="600" >

## Conflict Entity Categories

The dataset categorizes visual-semantic conflicts into three hierarchical levels based on semantic plausibility and occurrence frequency:

| Category | Example Entities | Context & Risk Description |
| :--- | :--- | :--- |
| **Common Mistakes**<br>*(High-Frequency Violations)* | `mobile phone`, `bread`, `coffee`, `fruit`, `food` | **Everyday Negligence:** Items strictly prohibited but plausibly introduced due to human error. Poses subtle but significant sterility and hygiene risks. |
| **Occasional Mistakes**<br>*(Environmental Breaches)* | `ant`, `butterfly`, `cat`, `dog`, `insect`, `small animal`, `plant`, `candle` | **Environmental Control Lapses:** Rare but severe breaches involving biological contaminants (flora/fauna) or fire hazards. Represents "out-of-context" living threats to the sterile field. |
| **Unreasonable Conflicts**<br>*(Semantic Absurdities)* | `chef`, `balloon`, `Teddy Bear`, `toy`, `No Parking sign` | **Logical Impossibilities:** Entities that defy the fundamental logic of a surgical suite. Used to stress-test the model's world knowledge and resistance to hallucinations. |

## Key Features

| Dataset Component | Data Scale | Generation & Annotation Method |
| :--- | :--- | :--- |
| **Synthetic Set** | **26,992** images | Diffusion-based inpainting on 4D-OR frames + Ensemble Verification |
| **Validation Set** | **509** images | Expert-annotated by medical professionals |

---
# 💻 VS-KC code

## Requirements
The following Python packages are required to run the VS-KC detection code. We recommend using Python 3.10.13 and CUDA 12.1 for optimal compatibility:

- torch==2.1.2                   
- qwen-vl-utils==0.0.10          
- transformers==4.51.3           
- diffusers==0.32.2              
- datasets==3.5.0                
- accelerate==1.7.0             
- peft==0.15.2                   

To install all dependencies:
```
pip install -r requirements.txt
```

## 🚀 Runing
The implementation executes five sequential stages:  
1. ![Generation](https://img.shields.io/badge/Stage-1_Generation-blue) `diffusion.py` - SD3.5 synthesis  
2. ![Filtering](https://img.shields.io/badge/Stage-2_Filtering-green) `llm-img.py` - Qwen-VL validation  
3. ![Evaluation](https://img.shields.io/badge/Stage-3_Evaluation-yellow) `acc-llm.py` - Baseline metrics  
4. ![Construction](https://img.shields.io/badge/Stage-4_Construction-orange) `DatasetJson.py` - Data formatting  
5. ![Fine-tuning](https://img.shields.io/badge/Stage-5_Finetuning-red) `Train_lora.py` - LoRA adaptation  

**📖 Complete documentation:** [test.txt](./VS-KC/test.txt)  


