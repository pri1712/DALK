from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate,LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
import numpy as np
import re
import string
from neo4j import GraphDatabase, basic_auth
import pandas as pd
from collections import deque
import itertools
from typing import Dict, List
import pickle
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import openai
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor
from langchain.llms import OpenAI
import os
from PIL import Image, ImageDraw, ImageFont
import csv
from gensim import corpora
from gensim.models import TfidfModel
from gensim.similarities import SparseMatrixSimilarity
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import sys
from time import sleep
from tqdm import tqdm
import datasets
import random
import logging
from together import Together

#import importlib.util
#file_path='/content/drive/MyDrive/code/DALK/KG4LLM/datasets_utils.py'
#spec = importlib.util.spec_from_file_location("datasets_utils", file_path)
#datasets_utils = importlib.util.module_from_spec(spec)
#spec.loader.exec_module(datasets_utils)

"""Setting up neo4j"""
uri =""
username = ""
password = ""



driver = GraphDatabase.driver(uri, auth=(username, password))
session = driver.session()



api_key=''
client = Together(api_key=api_key)

re1 = r'The extracted entities are (.*?)<END>'
re2 = r"The extracted entity is (.*?)<END>"
re3 = r"<CLS>(.*?)<SEP>"

def prompt_extract_keyword(input_text):
    # Define the template for the prompt
    template = """
    There are some samples:

    ### Instruction:
    'Learn to extract entities from the following medical questions.'

    ### Input:
    <CLS>Doctor, I have been having discomfort and dryness in my vagina for a while now.
    I also experience pain during sex. What could be the problem and what tests do I need?<SEP>
    The extracted entities are

    ### Output:
    <CLS>Doctor, I have been having discomfort and dryness in my vagina for a while now.
    I also experience pain during sex. What could be the problem and what tests do I need?<SEP>
    The extracted entities are Vaginal pain, Vaginal dryness, Pain during intercourse<EOS>

    ### Instruction:
    'Learn to extract entities from the following medical answers.'

    ### Input:
    <CLS>Okay, based on your symptoms, we need to perform some diagnostic procedures
    to confirm the diagnosis. We may need to do a CAT scan of your head and an
    Influenzavirus antibody assay to rule out any other conditions. Additionally, we
    may need to evaluate you further and consider other respiratory therapy or physical
    therapy exercises to help you feel better.<SEP>
    The extracted entities are

    ### Output:
    <CLS>Okay, based on your symptoms, we need to perform some diagnostic procedures
    to confirm the diagnosis. We may need to do a CAT scan of your head and an
    Influenzavirus antibody assay to rule out any other conditions. Additionally, we
    may need to evaluate you further and consider other respiratory therapy or physical
    therapy exercises to help you feel better.<SEP>
    The extracted entities are CAT scan of head (Head ct), Influenzavirus antibody assay,
    Physical therapy exercises; manipulation; and other procedures, Other respiratory therapy<EOS>

    Try to output:
    ### Instruction:
    'Learn to extract entities from the following medical questions.'

    ### Input:
    <CLS>{input}<SEP>The extracted entities are

    ### Output:
    """

    # Create the prompt template with input placeholders
    prompt = PromptTemplate(
        template=template,
        input_variables=["input"]
    )

    # Priming the AI system ; precedes the human prompt in the input sequence to the chat.
    system_message_prompt = SystemMessagePromptTemplate(prompt=prompt)
    system_message_prompt.format(input=input_text)

    # Create a template for human input messages.
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    # Combine system and human prompts into a chat prompt.
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )


    chat_prompt_with_values = chat_prompt.format_prompt(
        input=input_text,
        text={}
    )

   
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
        messages=chat_prompt_with_values.to_messages(),  
        max_tokens=1024,
        temperature=0.7,
        )

    # Use regex to extract entities from the model's response
    question_kg = re.findall(re1, response)

    # Return the extracted entities
    return question_kg

def find_shortest_path(start_entity_name,end_entity_name,candidate_list,driver):
    global exist_entity 

    with driver.session() as session:
        result = session.run(
            """
            MATCH (start_entity:Entity {name: $start_entity_name}),
                  (end_entity:Entity {name: $end_entity_name})
            MATCH p = allShortestPaths((start_entity)-[*..5]->(end_entity))
            RETURN p
            """,
            start_entity_name=start_entity_name,
            end_entity_name=end_entity_name
        )

        paths_between_entity=[]
        short_path=0

        for record in result:
            path = record["p"]
            entities, relations = [], []

            for i in range(len(path.nodes)):
              node = path.nodes[i]
              entity_name = node["name"]
              entities.append(entity_name)

             
              if i < len(path.relationships):
                  relationship = path.relationships[i]
                  relation_type = relationship.type
                  relations.append(relation_type)

            path_str = ""
            for i, entity in enumerate(entities):
                
              entity = entity.replace("_", " ")

             
              if entity in candidate_list:
                short_path = 1
                exist_entity = entity
              path_str += entity

              if i < len(relations):
                    relation = relations[i].replace("_", " ")
                    path_str += f" -> {relation} -> "

            if short_path == 1:
                paths_between_entity = [path_str]
                break
            else:
                paths_between_entity.append(path_str)  
                exist_entity = {}      

      
        if len(paths_between_entity) > 5:
            paths_between_entity = sorted(paths_between_entity, key=len)[:5]

        
        try:
            return paths_between_entity, exist_entity
        except:
            return paths_between_entity, {}

def combine_lists(*lists):
    combinations = list(itertools.product(*lists))
    results = []
    for combination in combinations:
        new_combination = []
        for sublist in combination:
            if isinstance(sublist, list):
                new_combination += sublist
            else:
                new_combination.append(sublist)
        results.append(new_combination)
    return results

def entity_neighbors(entity_name,disease_flag):
    """
    retrieves neighboring entities of a given entity in a KG. It also
    filters out specific relationships based on the input flags and organizes results
    into two categories: general neighbors and diseases.
    """

    disease=[]
    query = """
        MATCH (e:Entity)-[r]->(n)
        WHERE e.name = $entity_name
        RETURN type(r) AS relationship_type,
        collect(n.name) AS neighbor_entities
    """
    res=session.run(query=query,entity_name=entity_name)

    general_neighbors_list=[]
    for record in res:
        rel_type=record['relationship_type']
        if disease_flag == 1 and rel_type == 'has_symptom':
            continue

        neighbors = record["neighbor_entities"]

        if "disease" in rel_type.replace("_", " "):
            disease.extend(neighbors)

        else:
            general_neighbors_list.append([
                entity_name.replace("_", " "),
                rel_type.replace("_", " "),
                ','.join([x.replace("_", " ") for x in neighbors])
            ])
    return general_neighbors_list,disease

def prompt_path_finding(path_input):
    template = """
    There are some knowledge graph path. They follow entity->relationship->entity format.
    \n\n
    {Path}
    \n\n
    Use the knowledge graph information.
    Try to convert them to natural language, respectively. Use single quotation marks for entity name and relation name.
    And name them as Path-based Evidence 1, Path-based Evidence 2,...\n\n

    Output:
    """

    prompt = PromptTemplate(
        template = template,
        input_variables = ["Path"]
    )

    """
    prompt to find the shortest path between 2 entities ; or the path at a distance of at max k hops from the start node"""
    system_message_prompt = SystemMessagePromptTemplate(prompt = prompt)
    system_message_prompt.format(Path = path_input)

    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
    chat_prompt_with_values = chat_prompt.format_prompt(Path = path_input,text={})

    messages = []
    for msg in chat_prompt_with_values.to_messages():
        role = "system" if isinstance(msg, SystemMessage) else "user"
        messages.append({"role": role, "content": msg.content})

    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
        messages=messages,
        max_tokens=1024,
        temperature=0.7,
    )
    print("returning response \n")
    try:
        content = response.choices[0].message.content
        return content  # Return only the relevant text content
    except (IndexError, KeyError) as e:
        print(f"Error extracting content: {e}")
        return None  

def prompt_neighbor(neighbor):
    template = """
    There are some knowledge graph. They follow entity->relationship->entity list format.
    \n\n
    {neighbor}
    \n\n
    Use the knowledge graph information. Try to convert them to natural language, respectively.
    Use single quotation marks for entity name and relation name.
    And name them as Neighbor-based Evidence 1, Neighbor-based Evidence 2,...\n\n

    Output:
    """

    prompt = PromptTemplate(
        template = template,
        input_variables = ["neighbor"]
    )

    """prompt to find the neighbors of an entity ; so we can find the closest related items"""

    system_message_prompt = SystemMessagePromptTemplate(prompt = prompt)
    system_message_prompt.format(neighbor = neighbor)

    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
    chat_prompt_with_values = chat_prompt.format_prompt(neighbor = neighbor,\
                                                        text={})

    messages = []
    for msg in chat_prompt_with_values.to_messages():
        role = "system" if isinstance(msg, SystemMessage) else "user"
        messages.append({"role": role, "content": msg.content})

    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
        messages=messages,
        max_tokens=1024,
        temperature=0.7,
    )

    # print(response)
    print("returning response in neighbor \n")
    try:
        content = response.choices[0].message.content
        print(content)
        print("\n")
        return content  
    except (IndexError, KeyError) as e:
        print(f"Error extracting content: {e}")
        return None  

def self_aware_knowledge_retrieval(sub_graph, question):
    template = """
    There is a question and some knowledge graph. The knowledge graphs follow entity->relationship->entity list format.
    \n\n
    ##Graph: {graph}
    \n\n
    ##Question: {question}
    \n\n
    Please filter noisy and irrelevant knowledge from this knowledge graph that is useless or irrelevant to the give question.
    Output the filtered knowledges in the same format as the input knowledge graph.\n\n

    Filtered Knowledge:
    """

    prompt = PromptTemplate(
        template = template,
        input_variables = ["sub_graph", "question"]
    )

    system_message_prompt = SystemMessagePromptTemplate(prompt = prompt)
    system_message_prompt.format(graph = sub_graph, question=question)

    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
    chat_prompt_with_values = chat_prompt.format_prompt(graph = sub_graph, question=question,\
                                                        text={})

    messages = []
    for msg in chat_prompt_with_values.to_messages():
        role = "system" if isinstance(msg, SystemMessage) else "user"
        messages.append({"role": role, "content": msg.content})



    self_aware_subgraph = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
        messages=messages,
        max_tokens=1024,
        temperature=0.7,
    )
    print("returning response in self_aware \n")
    try:
        content = self_aware_subgraph.choices[0].message.content
        return content 
    except (IndexError, KeyError) as e:
        print(f"Error extracting content: {e}")
        return None  
        
def self_aware_knowledge_retrieval_ranking(sub_graph,question):
    template = """
    There is a question and some knowledge graph. The knowledge graphs follow entity->relationship->entity list format.
    \n\n
    ##Graph: {graph}
    \n\n
    ##Question: {question}
    \n\n
    Please rerank the knowledge graph and output at most 5
    important and relevant triples for solving the given question.
    Output the reranked knowledge in the following format:
    Reranked Triple1: xxx ——> xxx
    Reranked Triple2: xxx ——> xxx
    Reranked Triple3: xxx ——> xxx

    Answer:
    """

    """HERE K=5"""

    prompt = PromptTemplate(
        template = template,
        input_variables = ["sub_graph", "question"]
    )

    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    system_message_prompt = SystemMessagePromptTemplate(prompt = prompt)
    system_message_prompt.format(graph = sub_graph, question=question)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
    chat_prompt_with_values = chat_prompt.format_prompt(graph = sub_graph, question=question,\
                                                        text={})

    messages = []
    for msg in chat_prompt_with_values.to_messages():
        role = "system" if isinstance(msg, SystemMessage) else "user"
        messages.append({"role": role, "content": msg.content})

    reranked_knowledge_triplets = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
        messages=messages,
        max_tokens=1024,
        temperature=0.7,
    )

    return reranked_knowledge_triplets

def is_unable_to_answer(response):
    """Checks if the model is able to answer a given query confidently or not."""


    if isinstance(response, dict) and "choices" in response:
        prompt = response["choices"][0]["message"]["content"]
    elif isinstance(response, str):
        prompt = response
    else:
        raise ValueError("Invalid response format. Expected string or dict with 'choices'.")

    
    analysis = client.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
        prompt=prompt,  
        max_tokens=1,  
        temperature=0,
        n=1,
        presence_penalty=0.0,
        frequency_penalty=0.0,
    )

   
    score = analysis.choices[0].text.strip().replace("'", "").replace(".", "")

    if not score.isdigit():
        return True  

    threshold = 0.6
    return float(score) <= threshold

def autowrap_text(text, font, max_width):
    """
    Wraps the given text within the specified font and maximum width.So that each line is wrapped around if necessary.
    """
    text_lines = []
    if font.getsize(text)[0] <= max_width:
        text_lines.append(text)
    else:
        words = text.split(' ')
        i = 0
        while i < len(words):
            line = ''
            while i < len(words) and font.getsize(line + words[i])[0] <= max_width:
                line = line + words[i] + ' '
                i += 1
            if not line:
                line = words[i]
                i += 1
            text_lines.append(line)
    return text_lines

def final_merged_answer(input_text,path_graph,neighbor_graph):

    """
    Merges the path based subgraph and neighbor based subgraph to generate a final graph.
    """

    if len(path_graph)==0:
        path_graph=''
    if len(neighbor_graph)==0:
        neighbor_graph=''

    messages  = [
                SystemMessage(content="You are an excellent AI assistant to answering the following question"),
                HumanMessage(content='Question: '+input_text[0]),
                AIMessage(content="You have some medical knowledge information in the following:\n\n" +  '###'+ path_graph + '\n\n' +
                          '###' + neighbor_graph),
                HumanMessage(content="Answer: Let's think step by step: ")
                                   ]

    message_dict = []
    for msg in messages:
        role = "system" if isinstance(msg, SystemMessage) else "user"
        message_dict.append({"role": role, "content": msg.content})



    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
        messages=message_dict,
        max_tokens=1024,
        temperature=0.7,
    )
    print("returning response in final_merged_answer \n")
    try:
        content = response.choices[0].message.content
        return content 
    except (IndexError, KeyError) as e:
        print(f"Error extracting content: {e}")
        return None  

def prompt_document(question,instruction):
    template = """
    You are an excellent AI doctor, and you can diagnose diseases and recommend medications based on the symptoms in the conversation.\n\n
    Patient input:\n
    {question}
    \n\n
    You have some medical knowledge information in the following:
    {instruction}
    \n\n
    What disease does the patient have? What tests should patient take to confirm the diagnosis?
    What recommened medications can cure the disease?
    """

    prompt = PromptTemplate(
        template = template,
        input_variables = ["question","instruction"]
    )


    """
     creates a structured prompt
     for an AI model to analyze a patient's symptoms and provide a medical diagnosis,
     suggested tests, and medication recommendations
    """
    system_message_prompt = SystemMessagePromptTemplate(prompt = prompt)
    system_message_prompt.format(question = question,
                                 instruction = instruction)

    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
    chat_prompt_with_values = chat_prompt.format_prompt(question = question,\
                                                        instruction = instruction,\
                                                        text={})

    messages = []
    for msg in chat_prompt_with_values.to_messages():
        role = "system" if isinstance(msg, SystemMessage) else "user"
        messages.append({"role": role, "content": msg.content})


    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
        messages=messages,
        max_tokens=1024,
        temperature=0.7,
    )
    try:
        content = response.choices[0].message.content
        return content 
    except (IndexError, KeyError) as e:
        print(f"Error extracting content: {e}")
        return None  

""" This file formats prompts based on the dataset ; as different datasets are formatted differently. It also computes the accuracy of the prediction. """
import datasets
import random
random.seed(45)
import os
import json
import time
import logging

class Dataset_parent:

    def __init__(self):
        self.template_ner='''Extract all the biomedicine-related entity from the following question and choices, output each entity in a single line with a serial number (1., 2., ...)
        Question: {}
        The extracted entities are:
        '''
        self.template = '''Question: {}
        Answer: The option is: '''
        self.template_CoT = '''Question: {}
        Answer: Let's think step by step. '''
        self.template_inference = '''Question: {}
        Answer: Let's think step by step. {} Therefore, the letter option (only the letter) is:'''

    def load_dataset(self):
        """Returns the processed dataset."""
        return self.data

    def load_original_dataset(self):
        """Returns the original unprocessed dataset."""
        return self.data_original

class medmcqaZeroshotsProcessor(Dataset_parent):
    """Processor for the MedMCQA dataset with zero-shot prompts."""

    def __init__(self):
        super().__init__()
        self.data=self._load_json('')
        self.data_original=self._load_json('')
        self.num2answer = {0: 'A', 1: 'B', 2: 'C',3: 'D'}

    @staticmethod
    def _load_json(path_to_file):
        try:
            return json.load(open(path_to_file))
        except Exception as e:
            logging.error(f"Error loading json file: {str(e)}")

    def generate_prompt_ner(self, item):
        question = self._format_question(item)
        return self.template_ner.format(question)

    def generate_prompt(self, item):
        return self._format_question(item)

    def _format_question(self, item):
        """Helper to format the question with answer options."""
        question = item['question']
        options = f"\nA. {item['opa']}\nB. {item['opb']}\nC. {item['opc']}\nD. {item['opd']}\n"
        return question + options

    def parse(self, ret, item):
        ret = ret.replace('.', '')
        if len(ret) > 1:
            ret = ret[0]

        item['prediction'] = ret
        correct_answer = self.num2answer[item['cop']]

        return item, int(correct_answer.strip() == ret.strip())


class medqaZeroshotsProcessor(Dataset_parent):
    """Processor for the MedQA dataset with zero-shot prompts."""
    def __init__(self):
        super().__init__()
        self.data=self._load_json('')
        self.data_original=self._load_json('')
        self.num2answer = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

    @staticmethod
    def _load_json(path_to_file):
        try:
            return json.load(open(path_to_file))
        except Exception as e:
            logging.error(f"Error loading json file: {str(e)}")

    def generate_prompt_ner(self, item):
        question = self._format_question(item)
        return self.template_ner.format(question)

    def generate_prompt(self, item):
        return self._format_question(item)

    def _format_question(self, item):
        """Helper to format the question with answer options."""
        question = item['question']
        options = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(item['choices'])])
        return question + "\n" + options + "\n"

    def parse(self, ret, item):
        ret = ret.replace('.', '')
        if len(ret) > 1:
            ret = ret[0]

        item['prediction'] = ret
        correct_answer = item['choices'].index(item['answer'][0])
        correct_letter = self.num2answer[correct_answer]

        return item, int(correct_letter.strip() == ret.strip())


class mmluZeroshotsProcessor(Dataset_parent):
    """Processor for the MMLU dataset with zero-shot prompts."""
    def __init__(self):
        super().__init__()
        self.data=self._load_json('')
        self.data_original=self._load_json('')
        self.num2answer = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

    @staticmethod
    def _load_json(path_to_file):
        try:
            return json.load(open(path_to_file))
        except Exception as e:
            logging.error(f"Error loading json file: {str(e)}")

    def generate_prompt_ner(self, item):
        question = self._format_question(item)
        return self.template_ner.format(question)

    def generate_prompt(self, item):
        return self._format_question(item)

    def _format_question(self, item):
        """Helper to format the question with answer options."""
        question = item['question']
        options = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(item['choices'])])
        return question + "\n" + options + "\n"

    def parse(self, ret, item):
        ret = ret.replace('.', '')
        if len(ret) > 1:
            ret = ret[0]

        item['prediction'] = ret
        correct_letter = self.num2answer[item['answer']]

        return item, int(correct_letter.strip() == ret.strip())

class qa4mreZeroshotsProcessor(Dataset_parent):

    def __init__(self):
        super().__init__()
        self.data=self._load_json('')
        self.data_original=self._load_json('')
        self.num2answer = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E'}

    @staticmethod
    def _load_json(path_to_file):
        try:
            return json.load(open(path_to_file))
        except Exception as e:
            logging.error(f"Error loading json file: {str(e)}")

    def generate_prompt_ner(self, item):
        question = self._format_question(item)
        return self.template_ner.format(question)

    def generate_prompt(self, item):
        return self._format_question(item)

    def _format_question(self, item):
        """Helper to format the question with answer options."""
        question = item['question_str']
        options = "\n".join([
            f"{chr(65+i)}. {ans}" for i, ans in enumerate(item['answer_options']['answer_str'])
        ])
        return question + "\n" + options + "\n"

    def parse(self, ret, item):
        ret = ret.replace('.', '')
        if len(ret) > 1:
            ret = ret[0]

        item['prediction'] = ret
        correct_answer_id = int(item['correct_answer_id'])
        correct_letter = self.num2answer[correct_answer_id]

        return item, int(correct_letter.strip() == ret.strip())

def serialize_for_json(obj):
    """Convert custom objects into dictionaries."""
    if hasattr(obj, '__dict__'):
        return obj.__dict__
    else:
        raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

dataset2processor = {
    'medmcqa': medmcqaZeroshotsProcessor,
    'medqa': medqaZeroshotsProcessor,
    'mmlu': mmluZeroshotsProcessor,
    'qa4mre': qa4mreZeroshotsProcessor
}
datasets_dir = ['qa4mre','medmcqa', 'mmlu']

def main():
    print("in main")

    """Setting up open ai"""




    """Building KG in Neo4j now"""
 
    # session.run("MATCH (n) DETACH DELETE n") 

    # df = pd.read_csv('',sep='\t',header=None,names=['head','relation','tail'])

    # for i,row in df.iterrows():
    #     head_name = row['head']
    #     tail_name = row['tail']
    #     relation_ht = row['relation']
    #     query = (
    #         "MERGE (h:Entity { name: $head_name }) " ##Merge is used to check if the given entity already exists in the KG, if not it creates it.
    #         "MERGE (t:Entity { name: $tail_name }) "
    #         "MERGE (h)-[r:`" + relation_ht + "`]->(t)"
    #     )

    #     try:
    #         session.run(query=query,head_name=head_name,tail_name=tail_name,relation_name=relation_ht)
    #     #session
    #     except Exception as e:
    #         logging.error(f"Error occured while constructing the KG: {str(e)}")
    #         continue


    """Keyword extraction using OPENAI from query ; the regex expressions are used to extract relevant strings from the returned text"""




    with open('code/DALK/output.csv', 'w', newline='') as f_out:
        writer=csv.writer(f_out)
        writer.writerow(['Question', 'Label', 'MindMap','GPT3.5','BM25_retrieval','Embedding_retrieval','KG_retrieval','GPT4'])

    with open('code/DALK/KG4LLM/Alzheimers/keyword_emb.pkl', 'rb') as f_key:
        keyword_embeddings_pkl = pickle.load(f_key)

    with open('code/DALK/KG4LLM/Alzheimers/entity_embeddings.pkl', 'rb') as f_ent:
        entity_embeddings_pkl = pickle.load(f_ent)

    for dataset in datasets_dir:
        processor = dataset2processor[dataset]()
        data = processor.load_dataset()

        acc, total_num = 0, 0
        generated_data = []

        for item in tqdm(data):
            """Extract entities from the questions"""
            input_text=[processor.generate_prompt(item)]
            entity_list=item['entity'].split('\n')
            query_entity=[]

            for entity in entity_list:
                try:
                    entity = entity.split('.')[1].strip()
                    query_entity.append(entity)
                except Exception as e:
                    logging.error(f"Error occured while extracting entities from the query : {str(e)}")

            similar_entity = []
            entity_embeddings_emb = pd.DataFrame(entity_embeddings_pkl["embeddings"])

            for kg_entity in query_entity:
                keyword_index = keyword_embeddings_pkl["keywords"].index(kg_entity)
                kg_entity_emb = np.array(keyword_embeddings_pkl["embeddings"][keyword_index])

                cos_similar = cosine_similarity(entity_embeddings_emb,kg_entity_emb.reshape(1,-1))
                max_index = cos_similar.argmax()

                find_unique = entity_embeddings_pkl["entities"][max_index]

                while find_unique in similar_entity:
                    cos_similar[max_index] = 0
                    max_index = cos_similar.argmax()
                    find_unique = entity_embeddings_pkl["entities"][max_index]

                """Adding only unique entities to the list of entities similar to Entity E."""

                similar_entity.append(find_unique)

            """" Extracting paths between the similar entities with k=5"""

            if len(similar_entity) > 1:
                start_entity=similar_entity[0]
                candidate_entity=similar_entity[1:]

                result_path_list=[]
                while True:
                    new_start_entity = 0
                    current_paths = []

                    while len(candidate_entity) > 0:
                        end_entity = candidate_entity[0]
                        candidate_entity.remove(end_entity)
                        paths, exist_entity = find_shortest_path(start_entity, end_entity, candidate_entity,driver)
                        path_list = []
                        if paths == [''] or paths == []: ## no path from start to end with at max 5 hops
                            new_start_entity = 1
                            if candidate_entity == []:
                                new_start_entity = 0
                                break
                            start_entity = candidate_entity[0]
                            candidate_entity.remove(start_entity)
                            break
                        else:
                            for p in paths:
                                path_list.append(p.split('->'))

                            if len(path_list) > 0:
                                current_paths.append(path_list)

                        if exist_entity!={}:
                            try:
                                candidate_entity.remove(exist_entity)
                            except:
                                logging.error(f"Error occured while extracting exist_entity: {str(e)}")
                                continue

                        start_entity = end_entity
                    result_path = combine_lists(*current_paths)

                    if len(result_path) > 0:
                        result_path_list.extend(result_path)

                    if new_start_entity == 1:
                        continue
                    else:
                        break


                start_tmp = []
                """Extracting unique starting points"""
                for path_new in result_path_list:
                    if len(path_new) == 0:
                        continue
                    if path_new[0] not in start_tmp:
                        start_tmp.append(path_new[0])

                if start_tmp==[]:
                    result_path = {}
                    single_path = {}

                else:
                    if len(start_tmp) == 1:
                        result_path = result_path_list[:5]
                    else:
                        result_path = []
                        if len(start_tmp) >= 5:
                            for path_new in result_path_list:
                                if path_new == []:
                                    continue
                                if path_new[0] in start_tmp:
                                    result_path.append(path_new)
                                    start_tmp.remove(path_new[0])
                                if len(result_path) == 5:
                                    break
                        else:
                            count_per_entity = 5 // len(start_tmp)
                            rem = 5 % len(start_tmp)
                            current_count = 0
                            for path_new in result_path_list:
                                if len(result_path) < 5:
                                    if path_new == []:
                                        continue
                                    if path_new[0] in start_tmp:
                                        if current_count < count_per_entity:
                                            result_path.append(path_new)
                                            current_count += 1
                                        else:
                                            start_tmp.remove(path_new[0])
                                            current_count = 0
                                            if path_new[0] in start_tmp:
                                                result_path.append(path_new)
                                                current_count += 1

                                        if len(start_tmp) == 1:
                                            count_per_entity = count_per_entity + rem
                                else:
                                    break
                    try:
                        single_path = result_path_list[0]
                    except:
                        single_path = result_path_list


            else:
                result_path = {}
                single_path = {}

            """ Neighbor based exploration in KG using entities from query , separating disease and non disease entities ,this
            may be for pruning purposes wherein irrelevant nodes are kept in check , otherwise we may not get disease relevant info"""

            neighbor_list=[]
            neighbor_disease_list=[]

            for entity in similar_entity:
                disease_flag = 0
                neighbors,disease = entity_neighbors(entity,disease_flag)
                neighbor_list.extend(neighbors)

                while len(disease):
                    new_disease = []
                    for disease_tmp in disease:
                        if disease_tmp in similar_entity:
                            new_disease.append(disease_tmp)

                    if len(new_disease) != 0:
                        for disease_entity in new_disease:
                            disease_flag = 1
                            print(f"Neighbor of {entity} is {disease_entity}, exploring {disease_entity} further")
                            neighbors,disease = entity_neighbors(disease_entity,disease_flag)
                            neighbor_disease_list.extend(neighbors)

                    else:
                        for disease_entity in disease:
                            disease_flag = 1
                            neighbors,disease = entity_neighbors(disease_entity,disease_flag)
                            neighbor_disease_list.extend(neighbors)
                    if len(neighbor_disease_list) > 10:
                        break

            if len(neighbor_list)<=5:
                neighbor_list.extend(neighbor_disease_list)


            """Forming prompts using entities in path based exploration found abovve and then ranking the answers received for removing noise"""

            if len(similar_entity) > 1:
                response_of_KG_list_path = []
                if result_path == {}:
                    response_of_KG_list_path = []
                    path_sampled = []
                else:
                    result_new_path = []
                    for total_path_i in result_path:
                        path_input = "->".join(total_path_i)
                        result_new_path.append(path_input)

                    path = "\n".join(result_new_path)
                    path_sampled = self_aware_knowledge_retrieval_ranking(path, input_text[0])

                    response_of_KG_list_path = prompt_path_finding(path_sampled)
                    if is_unable_to_answer(response_of_KG_list_path):
                        response_of_KG_list_path = prompt_path_finding(path_sampled)

            else:
                response_of_KG_list_path = '{}'

            response_single_path = prompt_path_finding(single_path)
            print(response_single_path)
            if is_unable_to_answer(response_single_path):
                response_single_path = prompt_path_finding(single_path)


            """Prompt generation using neighbor based exploration and then reranking based on relevance"""

            response_of_KG_list_neighbor = []
            neighbor_new_list = []
            for neighbor_i in neighbor_list:
                neighbor = "->".join(neighbor_i)
                neighbor_new_list.append(neighbor)
            if len(neighbor_new_list) > 5:
                neighbor_input = "\n".join(neighbor_new_list[:5])
            else:
                neighbor_input = "\n".join(neighbor_new_list)
            neighbor_input_sampled = self_aware_knowledge_retrieval_ranking(neighbor_input, input_text[0])
            response_of_KG_neighbor = prompt_neighbor(neighbor_input_sampled)
            if is_unable_to_answer(response_of_KG_neighbor):
                response_of_KG_neighbor = prompt_neighbor(neighbor_input)

            """Generating the final prompt using above 2 prompts and also calculating accuracy"""


            output_all = final_merged_answer(input_text[0], response_of_KG_list_path, response_of_KG_neighbor)


            if is_unable_to_answer(output_all):
                output_all = final_merged_answer(input_text[0], response_of_KG_list_path, response_of_KG_neighbor)


            ret_parsed, acc_item = processor.parse(output_all, item)

            ret_parsed.update({
                'path': path_sampled,
                'neighbor_input': neighbor_input_sampled,
                'response_of_KG_list_path': response_of_KG_list_path,
                'response_of_KG_neighbor': response_of_KG_neighbor
            })


            if ret_parsed['prediction'] in processor.num2answer.values():
                acc += acc_item
                total_num += 1


            generated_data.append(ret_parsed)


        print(dataset)
        print('accuracy:', acc / total_num)
        print("\n")
        try:
   
            json_data = json.dumps(generated_data, default=serialize_for_json, indent=4)

            with open('response.json', 'w') as f:
                f.write(json_data)

            print("JSON dump successful!")
        except TypeError as e:
            logging.error(f"Error in json dump: {str(e)}")

if __name__ == '__main__':
    main()
