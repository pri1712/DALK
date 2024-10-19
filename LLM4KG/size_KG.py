import numpy as np
from neo4j import GraphDatabase, basic_auth
import pandas as pd
import os
import csv

uri = ""
username = ""
password = ""

"""Check num of triples and nodes"""

driver = GraphDatabase.driver(uri, auth=(username, password))
session = driver.session()


query = "MATCH ()-[r]->() RETURN count(r) AS totalRelationships;"
result = session.run(query)

for record in result:
    print(f"Number of triples: {record['totalRelationships']}")
print("\n")

query = "MATCH (n) RETURN count(n) AS Nodes;"
result = session.run(query)

for record in result:
    print(f"Number of triples: {record['Nodes']}")


session.close()
driver.close()