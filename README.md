# RainbowGPT

<div align="center">
  <p>
    <a align="center" href="https://github.com/ZhuJD-China/RainbowGPT" target="_blank">
      <img width="20%" height="150"  src="https://github.com/ZhuJD-China/RainbowGPT/blob/master/imgs/logo.jpg"></a>
  </p>
</div>

## 🚀RainbowAgent Integration
📈RainbowGPT now includes a powerful Stock Analysis module, integrating various technologies to provide comprehensive insights into the stock market. 

⚡The RainbowGPT combines AI Agent proxy, GPT-4, GPT3.5, ChatGlm3, Qwen LLM, ChromaDB vector database, Langchain knowledge base question-answer retrieval, and the Google search engine.

<p align="center">
✨ <a href="https://github.com/openai/openai-cookbook" >Navigate at [cookbook.openai.com]</a> •  <br>
🦜️🔗 <a href="https://github.com/langchain-ai/langchain" > LangChain ⚡ Building applications with LLMs through composability ⚡</a>  •  <br>
🤗 <a href="https://huggingface.co/Qwen">Qwen HF</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/organization/qwen">Qwen ModelScope</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://arxiv.org/abs/2309.16609">Qwen Paper</a> &nbsp&nbsp ｜ &nbsp&nbsp🖥️ <a href="https://modelscope.cn/studios/qwen/Qwen-72B-Chat-Demo/summary">Qwen Demo</a>•  <br>
🤗 <a href="https://huggingface.co/THUDM/chatglm3-6b" target="_blank">Chatglm3 HF Repo</a> • 🤖 <a href="https://modelscope.cn/models/ZhipuAI/chatglm3-6b" target="_blank">Chatglm3 ModelScope</a> • 📃 <a href="https://arxiv.org/abs/2103.10360" target="_blank">[GLM@ACL 22]</a> <a href="https://github.com/THUDM/GLM" target="_blank">[GitHub]</a> • 📃 <a href="https://arxiv.org/abs/2210.02414" target="_blank">[GLM-130B@ICLR 23]</a> <a href="https://github.com/THUDM/GLM-130B" target="_blank">[GitHub]</a> <br>
</p>
<p align="center">

## Getting Started
### Free Use of GPT API
We will regularly expand capacity based on usage, and we will continue to provide the API for free as long as we are not officially sanctioned. If this project is helpful to you, please consider giving us a ***Star***.

This API Key is used for forwarding API requests. You need to change the Host to `api.chatanywhere.com.cn` (preferred for domestic usage) or `api.chatanywhere.cn` (for international usage, domestic users need a global proxy).

- 🚀 [Apply for a Free API Key in Beta](https://api.chatanywhere.org/v1/oauth/free/github/render)
- Forwarding Host1: `https://api.chatanywhere.com.cn` (Domestic relay, lower latency, recommended)
- Forwarding Host2: `https://api.chatanywhere.cn` (For international usage, domestic users need a global proxy)
- Check your balance and usage records (announcements are also posted here): [Balance Inquiry and Announcements](https://api.chatanywhere.org/)
- The forwarding API cannot directly make requests to the official api.openai.com endpoint. You need to change the request address to `api.chatanywhere.com.cn` to use it. Most plugins and software can be modified accordingly.

**Method 1**
```python
import openai
openai.api_base = "https://api.chatanywhere.com.cn/v1"
# openai.api_base = "https://api.chatanywhere.cn/v1"
```
**Method 2 (Use if Method 1 doesn't work)**
Modify the environment variable OPENAI_API_BASE. Search for how to change environment variables on your specific system. If changes to the environment variable don't take effect, restart your system.
```bash
OPENAI_API_BASE=https://api.chatanywhere.com.cn/v1
or OPENAI_API_BASE=https://api.chatanywhere.cn/v1
```
**Open Source gpt_academic**
Locate the `config.py` file and modify the `API_URL_REDIRECT` configuration to the following:
```python
API_URL_REDIRECT = {"https://api.openai.com/v1/chat/completions": "https://api.chatanywhere.com.cn/v1/chat/completions"}
# API_URL_REDIRECT = {"https://api.openai.com/v1/chat/completions": "https://api.chatanywhere.cn/v1/chat/completions"}
```

The free API Key has a limit of 60 requests per hour per IP address and Key. In other words, if you use multiple keys under the same IP, the total hourly request limit for all keys cannot exceed 60. Similarly, if you use a single key across multiple IPs, the hourly request limit for that key cannot exceed 60.


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


[![Star History Chart](https://api.star-history.com/svg?repos=ZhuJD-China/RainbowGPT&type=Timeline)](https://star-history.com/#ZhuJD-China/RainbowGPT&Timeline)
