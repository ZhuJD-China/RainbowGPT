# RainbowGPT

🚀 # RainbowAgent

RainbowAgent integrates AI Agent proxy, ChromaDB vector database, Langchain knowledge base question-answer retrieval, and Google search engine retrieval technologies.

## Knowledge Base QA Search Algorithm

The knowledge base QA search algorithm employs context compression as an optimization method for document retrieval. By using the context of the query, it reduces the document content through a document compressor, returning only information relevant to the query, thereby improving retrieval efficiency. The ensemble of retrievers combines the results of different retrievers to enhance overall performance.

# BM25 Algorithm

BM25 (Best Matching 25) is an improved TF-IDF algorithm commonly used in information retrieval tasks. It aims to capture the importance of words in a document by considering the impact of document length on weights. Here are the key points of the BM25 algorithm:

1. **Inverse Document Frequency (IDF):** Calculates the inverse document frequency of words to measure their importance. In BM25, IDF takes into account the distribution of words across the entire document collection.

2. **Document Length (\|D\|):** Considers the length of the document to adjust weights. Adjustments are made to account for the potential lower contribution of words in longer documents.

3. **Adjustment Parameters \(k_1\) and \(b\):** These parameters allow users to fine-tune the algorithm's performance. \(k_1\) controls the saturation of term frequency, and \(b\) controls the impact of document length.

4. **Term Frequency \(f(q_i, D)\):** Calculates the frequency of words in the document to determine their importance.

The BM25 algorithm is computed using the following formula:

\[ BM25(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avg\_dl}}\right)} \]

where \(q_i\) represents words in the query, \(D\) is the document, \(n\) is the number of words in the query, and \(\text{avg\_dl}\) is the average document length. BM25 calculates a matching score between the query and document, facilitating information retrieval.

Feel free to integrate BM25 into your information retrieval systems for improved search performance!


## BM25 Retrievers

- **BM25-based Retriever:** Specialized in finding relevant documents based on keywords, suitable for sparse retrieval.

- **Embedding Similarity Retriever:** Uses embedding vectors for document and query embedding, excelling in finding relevant documents based on semantic similarity, suitable for dense retrieval.

## Application

By combining the results of sparse and dense retrievals, RainbowAgent leverages the strengths of each to improve retrieval performance, achieving more comprehensive document matching.


👋 Retrieval Search
![Retrieval_Search](https://github.com/ZhuJD-China/RainbowGPT/blob/master/imgs/Retrieval_Search.png)

⚡ Summarization
![Summarization](https://github.com/ZhuJD-China/RainbowGPT/blob/master/imgs/Summarization.png)

🤖 Chatbots
![Chatbots](https://github.com/ZhuJD-China/RainbowGPT/blob/master/imgs/Chatbots.png)

📚 SQL Agent
![SQLAgent](https://github.com/ZhuJD-China/RainbowGPT/blob/master/imgs/SQLAgent.png)

🌐 Web Scraping
![WebScraping](https://github.com/ZhuJD-China/RainbowGPT/blob/master/imgs/WebScraping.png)

🤗 新增UI界面
![WebScraping](https://github.com/ZhuJD-China/RainbowGPT/blob/master/imgs/exp.png)
