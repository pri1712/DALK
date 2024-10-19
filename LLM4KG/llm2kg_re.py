import os
import time
import json
import logging
from tqdm import tqdm
from api_utils import *


years=[2011,2012] 


template_summary = """Read the following abstract, generate short summary about {} entity "{}" to illustrate what is {}'s 
relationship with other medical entity.
Abstract: {}
Summary: """


"""Above template is used to prompt an LLM to summarize info about a particular entity""" 

template_relation_extraction_ZeroCoT = '''
Read the following summary, answer the following question.
Summary: {}
Question: predict the relationship between {} entity "{}" and {} entity "{}", first choose from the following options:
{}
Answer: Let's think step by step: '''

"""Above template is used to prompt an LLM to predict the relationship between two entities"""

template_relation_extraction_ZeroCoT_answer = '''
Read the following summary, answer the following question.
Summary: {}
Question: predict the relationship between {} entity "{}" and {} entity "{}", first choose from the following options:
{}
Answer: Let's think step by step: {}. So the answer is:'''

"""Above template is used to prompt an LLM to finalize the relationship between two entities."""

entity_map = {
    "Species": "anatomies",
    "Chromosome": "cellular components",
    "CellLine": "cellular components",
    "SNP": "biological processes",
    "ProteinMutation":"biological processes",
    "DNAMutation":"biological processes",
    "ProteinAcidChange":"biological processes",
    "DNAAcidChange":"biological processes",
    "Gene": "genes",
    "Chemical": "compounds",
    "Disease": "diseases"
}

entities2relation = {
    ("genes", "genes"): ["covaries", "interacts", "regulates"],
    ("diseases", "diseases"): ["resembles"],
    ("compounds", "compounds") : ["resembles"],
    ("genes", "diseases"): ["downregulates","associates","upregulates"],
    ("genes", "compounds"): ["binds", "upregulates", "downregulates"],
    ("compounds", "diseases"): ["treats", "palliates"],
}


"""Map the relations that are possible between two entities"""

valid_type = ["genes", "compounds", "diseases"]
"""This list defines the entity types that are relevant for the relationship extraction task"""


def readliterature():
    
    """organizing litearature by year"""

    """
    Reads the pubtator files and stores the data in lit_by_year by using year as the index.
    
    Args:
        None
    Returns:
        A dictionary where the keys are years and the values are lists of dictionaries. 
        Each dictionary in the list represents a year's pubtator file and each dictionary contains the data such as title, abstract, 
        entity_id, entity_name, and entity_type from that file.
    """
    lit_by_year = {year: [] for year in years}

    for year in years:
        # Open the corresponding pubtator file for the year
        file_path = os.path.join('/home/priyanshu/Documents/NUS application/code/DALK/LLM4KG/by_year', f'{year}.pubtator')
        with open(file_path, 'r') as file:
            literature = {'entity': {}}
            for line in file.readlines():
                line = line.strip()
                if line == '' and literature!={}: #end of the literature.
                    for entity_id in literature['entity']:
                        literature['entity'][entity_id]['entity_name'] = list(literature['entity'][entity_id]['entity_name'])
                    lit_by_year[year].append(literature)
                    literature = {'entity': {}}
                    continue

                if '|t|' in line:
                    literature['title'] = line.split('|t|')[1]
                elif '|a|' in line:
                    literature['abstract'] = line.split('|a|')[1]
                else:
                    line_list = line.split('\t')
                    if len(line_list) != 6:
                        entity_name, entity_type, entity_id = line_list[3], line_list[4], None
                    else:
                        entity_name, entity_type, entity_id = line_list[3], line_list[4], line_list[5]
                    if entity_id == '-':
                        continue
                    if entity_id not in literature['entity']:
                        literature['entity'][entity_id] = {'entity_name':set(), 'entity_type': entity_type}
                    literature['entity'][entity_id]['entity_name'].add(entity_name)
            entity_type = set()

    return lit_by_year

def format_entity_name(entity_names):
    """
    Format a list of entity names into a single string.

    If the list contains only one entity, return that entity name.
    If the list contains multiple entities, return a string formatted as
    "{first_entity} ({remaining_entities})".

    Args:
        entity_names: A list of entity names.

    Returns:
        A formatted string containing the entity names.
    """
    if not entity_names:
        return "No entity available"


    first_entity = entity_names[0]
    remaining_entities = ", ".join(entity_names[1:])

    if remaining_entities:
        return f"{first_entity} ({remaining_entities})"
    else:
        return first_entity


def build_options(entity_relation):
    

    """
    Build formatted options and a dictionary mapping for a list of relations.

    Args:
        entity_relation (list): A list of relations between two entities.

    Returns:
        tuple: A tuple containing the formatted string of options and a dictionary mapping the option labels to the corresponding relations.
    """

    entity_relation_new = entity_relation + [
        'no-relation', 
        'others, please specify by generating a short predicate in 5 words'
    ]

    
    option_list = ['A. ', 'B. ', 'C. ', 'D. ', 'E. ']

    
    ret = ''
    option2relation = {}
    
    for r, o in zip(entity_relation_new, option_list):
        ret += o + r + '\n' 
        option2relation[o.strip()] = r 
    
    return ret.strip(), option2relation


def main():
    """
    Extract entities from the literature and use them as inputs to the llm to generate summary as well as predict the 
    relationship between two entities.
    """
    extracted=[]

    demonstration=json.load(open('/home/priyanshu/Documents/NUS application/code/DALK/LLM4KG/demonstration.json'))
    demonstration='\n\n'.join(demonstration)+'\n'

    lit_by_year = readliterature()
    for year,literature in lit_by_year.items():
        for lit in tqdm(literature):
            title = lit['title']
            abstract = lit['abstract']
            current_item={'title': title,'abstract': abstract, 'triplet':[] }

            for i, (entity1_id, entity1_info) in enumerate(lit['entity'].items()):
                entity1_names,entity1_type =entity1_info['entity_name'],entity1_info['entity_type']

                if entity1_type not in entity_map:
                    logging.error(f"entity1 not in entity_map")
                    continue

                entity1_type_kg=entity_map[entity1_type]

                if entity1_type_kg not in valid_type:
                    logging.error(f"entity1 type is invalid")
                    continue
               
                
                entity1_name = format_entity_name(entity1_names)
                summary_msg=template_summary.format(entity1_type,entity1_name,entity1_name,abstract)
                summary_msg_wrapped = f'''"""{summary_msg}"""'''

                #Generating summary of the entity using abstract.
                try:
                    palm_summary=palm_api_request(summary_msg_wrapped)
                except:
                    logging.error(f"Error in generating summary for {entity1_name}")
                    continue

                for j,(entity2_id,entity2_info) in enumerate(lit['entity'].items()):
                    if i==j:
                        continue

                    entity2_names,entity2_type=entity2_info['entity_name'],entity2_info['entity_type']

                    if entity2_type not in entity_map:
                        logging.error(f"entity2 not in entity_map")
                        continue

                    entity2_type_kg=entity_map[entity2_type]

                    if entity2_type_kg not in valid_type:
                        logging.error(f"entity2 type is invalid")
                        continue

                    if(entity1_type_kg,entity2_type_kg) not in entities2relation:
                        logging.error(f"relation between {entity1_type_kg} and {entity2_type_kg} not in entities2relation")
                        continue
                    
                    time.sleep(5) #to avoid exceeding rate limit

                    entity2_name=format_entity_name(entity2_names)
                    entities_relation=entities2relation[(entity1_type_kg,entity2_type_kg)]
                    options,option2relation=build_options(entities_relation)

                    relation_extraction_msg=template_relation_extraction_ZeroCoT.format(palm_summary,entity1_type,entity1_name,entity2_type,entity2_name,options)
                    relation_extraction_msg_wrapped=f'''"""{relation_extraction_msg}"""'''
                    try:
                        palm_rel_extr=palm_api_request(demonstration+relation_extraction_msg_wrapped)
                    except:
                        logging.error(f"Exception occurred after palm_rrel_extr: {str(e)}")
                        continue
                    
                    if len(palm_rel_extr)==0:
                        continue; #empty
                    
                    palm_answer_rel=template_relation_extraction_ZeroCoT_answer.format(palm_summary,entity1_type,entity1_name,entity2_type,entity2_name,options,palm_rel_extr)
                    palm_answer_rel_wrapped=f"""{palm_answer_rel}"""
                    try:
                        palm_rel_extr_answer=palm_api_request(demonstration+palm_answer_rel_wrapped)
                    except:
                        logging.error(f"Exception occurred after palm_rel_extr_answer: {str(e)}")
                        continue

                    if len(palm_rel_extr_answer)==0:
                        continue; #empty

                    find=False
                    is_generated=False

                    for option,relation in option2relation.items():
                        if option in palm_rel_extr_answer or option[0]==palm_rel_extr_answer[0] or relation in palm_rel_extr_answer:
                            if relation == 'others, please specify by generating a short predicate in 5 words':
                                if '.' in relation:
                                    relation=palm_rel_extr_answer.split('.')[1]
                                else:
                                    relation=palm_rel_extr_answer
                                is_generated=True
                            find=True
                            break
                        if find==False:
                            is_generated=True
                            relation=palm_rel_extr_answer
                            print('NOT MATCH:', palm_rel_extr_answer, option2relation)

                        current_item['triplet'].append({
                                'entity1':{
                                    'entity_name': entity1_names,
                                    'entity_type': entity1_type_kg,
                                    'entity_id': entity1_id
                                    },
                                'entity2':{
                                    'entity_name': entity2_names,
                                    'entity_type': entity2_type_kg,
                                    'entity_id': entity2_id
                                    },
                                'relation': relation,
                                'is_generated': is_generated
                            })

            extracted.append(current_item)
        
        with open('DALK/LLM4KG/extracted/{}.json'.format(year), 'w') as f:
            print("writing into json file")
            f.write(json.dumps(extracted, indent=2))



if __name__ == '__main__':
    # print(tqdm)
    main()
    


