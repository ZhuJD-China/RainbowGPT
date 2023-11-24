# RainbowGPT

🚀 # RainbowAgent

RainbowAgent integrates AI Agent proxy, ChromaDB vector database, Langchain knowledge base question-answer retrieval, and Google search engine retrieval technologies.

## Knowledge Base QA Search Algorithm

The knowledge base QA search algorithm employs context compression as an optimization method for document retrieval. By using the context of the query, it reduces the document content through a document compressor, returning only information relevant to the query, thereby improving retrieval efficiency. The ensemble of retrievers combines the results of different retrievers to enhance overall performance.

## BM25 Algorithm

BM25 is an improved TF-IDF algorithm that considers the impact of document length on weights, better capturing the importance of words in the document. The algorithm is expressed by the following formula:

\[BM25(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avg\_dl}}\right)}\]

Where:
- \(f(q_i, D)\) is the frequency of the word \(q_i\) in document \(D\).
- \(\text{avg\_dl}\) is the average document length.
- \(k_1\) and \(b\) are tuning parameters.
- \(\text{IDF}(q_i)\) is the inverse document frequency.

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
