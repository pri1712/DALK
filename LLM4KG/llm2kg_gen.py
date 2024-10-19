import os
import time
import json
from api_utils import *
from llm2kg_re import *
import logging
from tqdm import tqdm

template = '''Read the following abstract, extract the relationships between each entity.
You can choose the relation from: (covaries, interacts, regulates, resembles, downregulates, upregulates, associates, binds, treats, palliates), or generate a new predicate to describe the relationship between the two entities.
Output all the extract triples in the format of "head | relation | tail". For example: "Alzheimer's disease | associates | memory deficits"

Abstract: {}
Entity: {}
Output: '''

years = [2011]

"""Due to hardware as well as API limitations, I would not be able to use data from all the years to construct the knowledge graph. I am thus using
    only 2011 and 12 ; and the rest of the relations are from the official implementation."""


def read_literature():
    lit_by_year = {year: [] for year in years}
    for year in years:
        with open(os.path.join('/home/priyanshu/Documents/NUS application/code/DALK/LLM4KG/by_year', '{}.pubtator'.format(year))) as f:
            literature = {'entity': {}}
            for line in f.readlines():
                line = line.strip()
                if line == ''  and literature != {}:
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

def main():
    no_relation, with_relation = 0, 0
    lit_by_year = read_literature()

    for year, literatures in lit_by_year.items():
        extracted = []
        for literature in tqdm(literatures):
            # time.sleep(5) #avoiding rate limiting issues with API
            title, abstract = literature['title'], literature['abstract']
            item = {
                'title': title,
                'abstract': abstract,
                'triplet':[]
            }
            entity_names = ', '.join([format_entity_name(entity_info['entity_name']) for entity_info in literature['entity'].values()])
            msg = template.format(abstract, entity_names)
            msg_wrapped = f'''"""{msg}"""'''

            try:
                response = palm_api_request(msg_wrapped)
                time.sleep(5)
            except Exception as e:
                logging.error(f"Exception occurred: {str(e)}")

                continue
            if response == []:
                logging.error(f"empty response")
                continue
            for triple in response.split('\n'):
                if triple == '':
                    logging.error(f"empty triple")
                    continue
                try:
                    entity1, relation, entity2 = triple.split(' | ')
                except Exception as e:
                    logging.error(f"Exception occurred: {str(e)}")

                    continue
                item['triplet'].append({
                    'entity1': {
                        'entity_name': entity1,
                    },
                    'entity2': {
                        'entity_name': entity2,
                    },
                    'relation': relation,
                })

            extracted.append(item)
        
        os.makedirs('DALK/LLM4KG/extracted', exist_ok=True)
        with open('DALK/LLM4KG/extracted/{}_s2s.json'.format(year), 'w') as f:
            print("writing into json file")
            f.write(json.dumps(extracted, indent=2))

if __name__ == '__main__':
    main()