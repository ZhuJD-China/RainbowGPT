# RainbowGPT

🚀 # RainbowAgent

RainbowAgent integrates AI Agent proxy, ChromaDB vector database, Langchain knowledge base question-answer retrieval, and Google search engine retrieval technologies.

## Knowledge Base QA Search Algorithm

The knowledge base QA search algorithm employs context compression as an optimization method for document retrieval. By using the context of the query, it reduces the document content through a document compressor, returning only information relevant to the query, thereby improving retrieval efficiency. The ensemble of retrievers combines the results of different retrievers to enhance overall performance.

## BM25 Algorithm

BM25 is an improved TF-IDF algorithm that considers the impact of document length on weights, better capturing the importance of words in the document. The algorithm is expressed by the following formula:

![BM25 Formula](https://latex.codecogs.com/svg.latex?BM25(D,%20Q)%20=%20\sum_{i=1}^{n}%20\text{IDF}(q_i)%20\cdot%20\frac{f(q_i,%20D)%20\cdot%20(k_1%20+%201)}{f(q_i,%20D)%20+%20k_1%20\cdot%20\left(1%20-%20b%20+%20b%20\cdot%20\frac{|D|}{\text{avg_dl}}\right)})

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
