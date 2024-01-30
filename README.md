
## About
* This repository consists of codes, models and data for:

"Cell-free biosynthesis combined with deep learning accelerates de novo-development of antimicrobial peptides"
[Nature Communications, 2023](https://www.nature.com/articles/s41467-023-42434-9) 

> Abstract:
Bioactive peptides are key molecules in health and medicine. Deep learning holds a big promise for the discovery and design of bioactive peptides. Yet, suitable experimental approaches are required to validate candidates in high throughput and at low cost. Here, we established a cell-free protein synthesis (CFPS) pipeline for the rapid and inexpensive production of antimicrobial peptides (AMPs) directly from DNA templates. To validate our platform, we used deep learning to design thousands of AMPs de novo. Using computational methods, we prioritized 500 candidates that we produced and screened with our CFPS pipeline. We identified 30 functional AMPs, which we characterized further through molecular dynamics simulations, antimicrobial activity and toxicity. Notably, six de novo-AMPs feature broad-spectrum activity against multidrug-resistant pathogens and do not develop bacterial resistance. Our work demonstrates the potential of CFPS for production and testing of bioactive peptides within less than 24 hours and <10$ per screen.

## Setup


- Clone this repo to your local machine using git clone
- Change the directory to the cloned repo
- Optionally build a new virtual environment
- Run pip install -r requirements.txt
- For a quick glance at the whole workflow run
 ```
 python example.py
 ```

## Supporting information for the paper

30AMPs.txt is the sequence of functional de-novo peptides discovered and characterized in this work in FASTA format.
500gen.txt is the sequence of peptides we built in the lab.
biochem folder contains data for supplementary figures X.
train_data folder contains data we used for model training. 
