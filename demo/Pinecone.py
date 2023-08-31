import openai
from typing import List, Iterator
import pandas as pd
import numpy as np
import os
import wget
from ast import literal_eval

# I've set this to our new embeddings model, this can be changed to the embedding model of your choice
EMBEDDING_MODEL = "text-embedding-ada-002"

# Ignore unclosed SSL socket warnings - optional in case you get these errors
import warnings

warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

"""Load data
In this section we'll load embedded data that we've prepared previous to this session."""

embeddings_url = 'https://cdn.openai.com/API/examples/data/vector_database_wikipedia_articles_embedded.zip'

# The file is ~700 MB so this will take some time
wget.download(embeddings_url)
import zipfile

with zipfile.ZipFile("vector_database_wikipedia_articles_embedded.zip", "r") as zip_ref:
    zip_ref.extractall("../data")
article_df = pd.read_csv('../data/vector_database_wikipedia_articles_embedded.csv')

article_df.head()
# Read vectors from strings back into a list
article_df['title_vector'] = article_df.title_vector.apply(literal_eval)
article_df['content_vector'] = article_df.content_vector.apply(literal_eval)

# Set vector_id to be a string
article_df['vector_id'] = article_df['vector_id'].apply(str)
article_df.info(show_counts=True)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import pinecone
import os

pinecone.init(api_key="f704e30a-bede-474d-914c-b80d42dcd3bc", environment="asia-southeast1-gcp-free")

pinecone.create_index("quickstart", dimension=8, metric="euclidean")

pinecone.list_indexes()
# Returns:
# ['quickstart']


index = pinecone.Index("quickstart")

# Upsert sample data (5 8-dimensional vectors)
index.upsert([
    ("A", [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
    ("B", [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]),
    ("C", [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]),
    ("D", [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]),
    ("E", [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
])

index.describe_index_stats()
# Returns:
# {'dimension': 8, 'index_fullness': 0.0, 'namespaces': {'': {'vector_count': 5}}}


index.query(
    vector=[0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
    top_k=3,
    include_values=True
)

pinecone.delete_index("quickstart")
