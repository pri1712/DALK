# DALK: Dynamic Co-Augmentation of LLMs and KGs for Alzheimer’s Disease Research
This repository contains the implementation of the DALK framework, which dynamically integrates Large Language Models (LLMs) and Knowledge Graphs (KGs) to improve query responses related to Alzheimer’s Disease (AD). The project uses open-source LLaMA models and Neo4j for knowledge graph storage. LLaMa is used for generating answers to given queries and Google Gemini is used to extract entities from the given annotated data.

## Overview:

DALK aims to enhance:  
  	1. LLM outputs using knowledge graphs for more accurate and context-relevant query responses on long tail knowledge and in particular about Alzheimers disease.  
    2. Knowledge graph construction using LLMs through entity and relation extraction from unstructured data.

## Setup Instructions:

1. **Cloning git repo**:  
```bash
git clone https://github.com/pri1712/DALK.git
cd DALK
```
2. **Installing required dependancies** in a virtual env/conda env:  
    Setup the env and install the requirements.txt in the following fashion;
```
python3 -m venv name_of_venv
source /bin/activate/name_of_venv
pip install -r requirements.txt
```
3. Open the `mindmap.py` file and **replace the placeholder text with your own API keys** to enable access to necessary services.  
4. Make a new **blank sandbox** on [Neo4j](https://sandbox.neo4j.com/):  
		This is necessary so as to store your nodes and relations.  
		Ps: if you want to see a cool visualization of your KG have a look at [this](https://neo4j.com/developer-blog/visualize-graph-embedding-algorithm-result-in-neuler/)
5. **LLM4KG Module**:  
   Ensure that the `run.sh` script has executable permissions. You can do this by running `chmod +x run.sh` in your terminal.
   
   Once done, execute the script with the required input file to extract entities into a JSON file (Due to API rate limiting constraints I was able to extract entities for only 2 of the files.)
6. **KG4LLM Module**:  
Navigate to the `DALK/KG4LLM` directory and run the `mindmap.py` script to run the LLM and ensure that it uses KG data.Ensure that all dependencies are installed and API keys are properly configured in `mindmap.py` before executing the script.  
	To run the script, use the following command in your terminal:  
	 ```bash
	 python3 mindmap.py
	 ```

## Credits:

- https://github.com/David-Li0406/DALK ; I used a lot of their boilerplate code as well as the entities for the files I was not able to extract data due to API rate limiting.
- https://github.com/wyl-willing/MindMap ; Used code as a reference for KG usage with LLMs.
- Together AI ; for their free 5$ worth of API credits :)
  
