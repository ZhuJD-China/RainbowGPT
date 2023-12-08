# RainbowGPT

<div align="center">
  <p>
    <a align="center" href="https://github.com/ZhuJD-China/RainbowGPT" target="_blank">
      <img width="20%" height="150"  src="https://github.com/ZhuJD-China/RainbowGPT/blob/master/imgs/logo.jpg"></a>
  </p>
</div>

## 🌈✨RainbowAgent Integration
📈RainbowGPT now includes a powerful Stock Analysis module, integrating various technologies to provide comprehensive insights into the stock market. 

⚡The RainbowGPT combines AI Agent proxy, GPT-4, GPT3.5, ChatGlm3, Qwen LLM, ChromaDB vector database, Langchain knowledge base question-answer retrieval, and the Google search engine.

<p align="center">
✨ <a href="https://github.com/openai/openai-cookbook" >Navigate at [cookbook.openai.com]</a> •  <br>
🦜️🔗 <a href="https://github.com/langchain-ai/langchain" > LangChain ⚡ Building applications with LLMs through composability ⚡</a>  •  <br>
🤗 <a href="https://huggingface.co/Qwen">Qwen HF</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/organization/qwen">Qwen ModelScope</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://arxiv.org/abs/2309.16609">Qwen Paper</a> &nbsp&nbsp ｜ &nbsp&nbsp🖥️ <a href="https://modelscope.cn/studios/qwen/Qwen-72B-Chat-Demo/summary">Qwen Demo</a>•  <br>
🤗 <a href="https://huggingface.co/THUDM/chatglm3-6b" target="_blank">Chatglm3 HF Repo</a> • 🤖 <a href="https://modelscope.cn/models/ZhipuAI/chatglm3-6b" target="_blank">Chatglm3 ModelScope</a> • 📃 <a href="https://arxiv.org/abs/2103.10360" target="_blank">[GLM@ACL 22]</a> <a href="https://github.com/THUDM/GLM" target="_blank">[GitHub]</a> • 📃 <a href="https://arxiv.org/abs/2210.02414" target="_blank">[GLM-130B@ICLR 23]</a> <a href="https://github.com/THUDM/GLM-130B" target="_blank">[GitHub]</a> <br>
</p>
<p align="center">

## ✨ Getting Started
### Environment
please make sure you have set up the environment and installed the required packages. Make sure you meet the above requirements, and then install the dependent libraries.
```bash
pip install -r requirements.txt
```
 **Markdown Tip:** When running `RainbowGPT_Agent_V2.2.py`, don't forget to relocate `3rd_modify/langchain_vectorstores_chroma.py` to your Langchain module! 🌈✨

### 🌐 Free Use of GPT API
We are committed to expanding capacity based on usage and providing the API for free as long as we are not officially sanctioned. If you find this project helpful, please consider giving us a ⭐.

This API Key is used for forwarding API requests. Change the Host to `api.chatanywhere.com.cn` (preferred for domestic usage) or `api.chatanywhere.cn` (for international usage, domestic users need a global proxy).

- 🚀 [Apply for a Free API Key in Beta](https://api.chatanywhere.org/v1/oauth/free/github/render)
- Forwarding Host1: `https://api.chatanywhere.com.cn` (Domestic relay, lower latency, recommended)
- Forwarding Host2: `https://api.chatanywhere.cn` (For international usage, domestic users need a global proxy)
- Check your balance and usage records (announcements are also posted here): [Balance Inquiry and Announcements](https://api.chatanywhere.org/)
- The forwarding API cannot directly make requests to the official api.openai.com endpoint. Change the request address to `api.chatanywhere.com.cn` to use it. Most plugins and software can be modified accordingly.

**Method 1**
```python
import openai
openai.api_base = "https://api.chatanywhere.com.cn/v1"
# openai.api_base = "https://api.chatanywhere.cn/v1"
```
**Method 2 (Use if Method 1 doesn't work)**
Modify the environment variable `OPENAI_API_BASE`. Search for how to change environment variables on your specific system. If changes to the environment variable don't take effect, restart your system.
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

The free API Key has a limit of 60 requests per hour per IP address and Key. If you use multiple keys under the same IP, the total hourly request limit for all keys cannot exceed 60. Similarly, if you use a single key across multiple IPs, the hourly request limit for that key cannot exceed 60.

## 🧠 Knowledge Base QA Search Algorithm
The knowledge base QA search algorithm optimizes document retrieval through context compression. Leveraging the query context, it strategically reduces document content using a document compressor, enhancing retrieval efficiency by returning only information relevant to the query. The ensemble of retrievers combines diverse results, creating a synergy that elevates overall performance.

## 🎯 BM25 Retrievers
- **BM25-based Retriever:** Specialized in efficiently locating relevant documents based on keywords, making it particularly effective for sparse retrieval.
- **Embedding Similarity Retriever:** Utilizes embedding vectors for document and query embedding, excelling in identifying relevant documents through semantic similarity. This retriever is well-suited for dense retrieval scenarios.

## 🚀 EnsembleRetriever
EnsembleRetriever is a powerful retrieval mechanism that combines the strengths of various retrievers. It takes a list of retrievers as input, integrates their results using the `get_relevant_documents()` methods, and reranks the outcomes using the Reciprocal Rank Fusion algorithm.

By leveraging the diverse strengths of different algorithms, EnsembleRetriever achieves superior performance compared to individual retrievers.

## Common Usage Pattern
🔄 The most effective use of the Knowledge Base QA Search involves combining a sparse retriever (e.g., BM25) with a dense retriever (e.g., embedding similarity). This "hybrid search" optimally utilizes the complementary strengths of both retrievers for comprehensive Knowledge.

📊 Explore the Stock Analysis module and unlock valuable insights for your investment decisions! 🚀 #StockAnalysis #RainbowGPT #AIInvesting

# RainbowGPT Overview


### Retrieval Search  |  Web Scraping Summarization
<img src="https://github.com/ZhuJD-China/RainbowGPT/blob/master/imgs/Retrieval_Search.png" width="400"/>  <img src="https://github.com/ZhuJD-China/RainbowGPT/blob/master/imgs/Summarization.png" width="400"/>

### Chatbots  |  SQL Agent
<img src="https://github.com/ZhuJD-China/RainbowGPT/blob/master/imgs/Chatbots.png" width="400"/>  <img src="https://github.com/ZhuJD-China/RainbowGPT/blob/master/imgs/SQLAgent.png" width="400"/>


👋 **Retrieval Search**
![Retrieval_Search](https://github.com/ZhuJD-China/RainbowGPT/blob/master/imgs/Retrieval_Search.png)

⚡🌐 **Web Scraping Summarization**
![Summarization](https://github.com/ZhuJD-China/RainbowGPT/blob/master/imgs/Summarization.png)

🤖 **Chatbots**
![Chatbots](https://github.com/ZhuJD-China/RainbowGPT/blob/master/imgs/Chatbots.png)

📚 **SQL Agent**
![SQLAgent](https://github.com/ZhuJD-China/RainbowGPT/blob/master/imgs/SQLAgent.png)


🤗 **Rainbow Agent UI**
![WebScraping](https://github.com/ZhuJD-China/RainbowGPT/blob/master/imgs/exp.png)

📊 **StockGPT Analysis**
![StockGPT](https://github.com/ZhuJD-China/RainbowGPT/blob/master/imgs/stock.png)

[![Star History Chart](https://api.star-history.com/svg?repos=ZhuJD-China/RainbowGPT&type=Timeline)](https://star-history.com/#ZhuJD-China/RainbowGPT&Timeline)

🚀⚡ Explore the diverse capabilities of RainbowGPT and leverage its powerful modules for your projects! 🌈✨

