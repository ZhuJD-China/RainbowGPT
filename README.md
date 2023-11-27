# RainbowGPT

<div align="center">
  <p>
    <a align="center" href="https://github.com/ZhuJD-China/RainbowGPT" target="_blank">
      <img width="20%" height="150"  src="https://github.com/ZhuJD-China/RainbowGPT/blob/master/imgs/logonew.jpg"></a>
  </p>
</div>

🚀 **RainbowAgent Integration**

RainbowAgent seamlessly integrates various technologies, including an AI Agent proxy, ChromaDB vector database, Langchain knowledge base question-answer retrieval, and the Google search engine.

## Knowledge Base QA Search Algorithm

The knowledge base QA search algorithm optimizes document retrieval through context compression. Leveraging the query context, it strategically reduces document content using a document compressor. This enhances retrieval efficiency by returning only information relevant to the query. The ensemble of retrievers combines diverse results, creating a synergy that elevates overall performance.

## BM25 Retrievers

- **BM25-based Retriever:** Specialized in efficiently locating relevant documents based on keywords, making it particularly effective for sparse retrieval.

- **Embedding Similarity Retriever:** Utilizes embedding vectors for document and query embedding, excelling in identifying relevant documents through semantic similarity. This retriever is well-suited for dense retrieval scenarios.

# EnsembleRetriever

EnsembleRetriever is a powerful retrieval mechanism that combines the strengths of various retrievers. It takes a list of retrievers as input, integrates their results using the `get_relevant_documents()` methods, and reranks the outcomes using the Reciprocal Rank Fusion algorithm.

By leveraging the diverse strengths of different algorithms, EnsembleRetriever achieves superior performance compared to individual retrievers.

## Common Usage Pattern

The most effective use of EnsembleRetriever involves combining a sparse retriever, such as BM25, with a dense retriever, like embedding similarity. This approach, known as "hybrid search," optimally utilizes the complementary strengths of both retrievers. The sparse retriever excels in finding relevant documents based on keywords, while the dense retriever is proficient in identifying relevant documents through semantic similarity.

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
