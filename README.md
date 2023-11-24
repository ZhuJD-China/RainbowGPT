# RainbowGPT

🚀 RainbowAgent集成AI Agent代理、ChromaDB向量数据库、Langchain知识库问答检索、Google搜索引擎检索等技术

其中知识库问答搜索算法是：上下文压缩是一种优化文档检索的方法，通过使用查询的上下文，通过文档压缩器减少文档内容，只返回与查询相关的信息，提高检索效率。
检索器合奏则是通过结合不同检索器的结果来提高整体性能。

BM25是一种改进的TF-IDF算法，它通过考虑文档长度对权重的影响，更好地捕捉了词在文档中的重要性。其算法公式如下：
···latex
BM25(D, Q) = ∑_{i=1}^{n} IDF(q_i) * (f(q_i, D) * (k_1 + 1)) / (f(q_i, D) + k_1 * (1 - b + b * |D| / avg_dl))
···
其中，f(q_i, D) 是词 q_i 在文档 D 中的频率，avg_dl 是平均文档长度，k_1 和 b 是调节参数，IDF(q_i) 是逆文档频率。

BM25算法的检索器： 基于BM25算法的检索器，擅长根据关键词查找相关文档，适用于稀疏检索。

嵌入相似性检索器： 使用嵌入向量进行文档和查询的嵌入，擅长根据语义相似性查找相关文档，适用于密集检索。

应用： 通过组合稀疏检索和密集检索的结果，充分发挥各自的优势，提高检索性能，实现更全面的文档匹配。

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
