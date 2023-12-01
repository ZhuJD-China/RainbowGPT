from dotenv import load_dotenv
from langchain.utilities.serpapi import SerpAPIWrapper
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain.utilities import ArxivAPIWrapper

from typing import Dict, Tuple
import os
import json

from langchain_experimental.tools import PythonAstREPLTool
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

load_dotenv()


class QwenLangChainIntegration:
    def __init__(self):
        self.search_tool = SerpAPIWrapper()
        self.wolfram_tool = WolframAlphaAPIWrapper()
        self.arxiv_tool = ArxivAPIWrapper()
        self.python_tool = PythonAstREPLTool()

        self.tools = [
            {
                'name_for_human': 'google search',
                'name_for_model': 'Search',
                'description_for_model': 'useful for when you need to answer questions about current events.',
                'parameters': [
                    {"name": "query", "type": "string", "description": "search query of google", 'required': True}
                ],
                'tool_api': self.tool_wrapper_for_qwen(self.search_tool)
            },
            {
                'name_for_human': 'Wolfram Alpha',
                'name_for_model': 'Math',
                'description_for_model': 'Useful for when you need to answer questions about Math, Science, Technology, Culture, Society and Everyday Life.',
                'parameters': [
                    {"name": "query", "type": "string", "description": "the problem to solved by Wolfram Alpha",
                     'required': True}
                ],
                'tool_api': self.tool_wrapper_for_qwen(self.wolfram_tool)
            },
            {
                'name_for_human': 'arxiv',
                'name_for_model': 'Arxiv',
                'description_for_model': 'A wrapper around Arxiv.org Useful for when you need to answer questions about Physics, Mathematics, Computer Science, Quantitative Biology, Quantitative Finance, Statistics, Electrical Engineering, and Economics from scientific articles on arxiv.org.',
                'parameters': [
                    {"name": "query", "type": "string", "description": "the document id of arxiv to search",
                     'required': True}
                ],
                'tool_api': self.tool_wrapper_for_qwen(self.arxiv_tool)
            },
            {
                'name_for_human': 'python',
                'name_for_model': 'python',
                'description_for_model': "A Python shell. Use this to execute python commands. When using this tool, sometimes output is abbreviated - Make sure it does not look abbreviated before using it in your answer. "
                                         "Don't add comments to your python code.",
                'parameters': [
                    {"name": "query", "type": "string", "description": "a valid python command.", 'required': True}
                ],
                'tool_api': self.tool_wrapper_for_qwen(self.python_tool)
            }
        ]

        self.checkpoint = "Qwen/Qwen-7B-Chat"
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.checkpoint, device_map="auto",
                                                          trust_remote_code=True).eval()
        self.model.generation_config = GenerationConfig.from_pretrained(self.checkpoint, trust_remote_code=True)
        self.model.generation_config.do_sample = False

    def tool_wrapper_for_qwen(self, tool):
        def tool_(query):
            query = json.loads(query)["query"]
            return tool.run(query)

        return tool_

    def build_planning_prompt(self, tools, query):
        tool_descs = []
        tool_names = []
        for info in tools:
            tool_descs.append(
                self.TOOL_DESC.format(
                    name_for_model=info['name_for_model'],
                    name_for_human=info['name_for_human'],
                    description_for_model=info['description_for_model'],
                    parameters=json.dumps(
                        info['parameters'], ensure_ascii=False),
                )
            )
            tool_names.append(info['name_for_model'])
        tool_descs = '\n\n'.join(tool_descs)
        tool_names = ','.join(tool_names)

        prompt = self.REACT_PROMPT.format(tool_descs=tool_descs, tool_names=tool_names, query=query)
        return prompt

    def parse_latest_plugin_call(self, text: str) -> Tuple[str, str]:
        i = text.rfind('\nAction:')
        j = text.rfind('\nAction Input:')
        k = text.rfind('\nObservation:')
        if 0 <= i < j:
            if k < j:
                text = text.rstrip() + '\nObservation:'
                k = text.rfind('\nObservation:')
        if 0 <= i < j < k:
            plugin_name = text[i + len('\nAction:'):j].strip()
            plugin_args = text[j + len('\nAction Input:'):k].strip()
            return plugin_name, plugin_args
        return '', ''

    def use_api(self, tools, response):
        use_toolname, action_input = self.parse_latest_plugin_call(response)
        if use_toolname == "":
            return "no tool founds"

        used_tool_meta = list(filter(lambda x: x["name_for_model"] == use_toolname, tools))
        if len(used_tool_meta) == 0:
            return "no tool founds"

        api_output = used_tool_meta[0]["tool_api"](action_input)
        return api_output

    def main(self, query, choose_tools):
        prompt = self.build_planning_prompt(choose_tools, query)
        print(prompt)
        stop = ["Observation:", "Observation:\n"]
        react_stop_words_tokens = [self.tokenizer.encode(stop_) for stop_ in stop]
        response, _ = self.model.chat(self.tokenizer, prompt, history=None, stop_words_ids=react_stop_words_tokens)

        while "Final Answer:" not in response:
            api_output = self.use_api(choose_tools, response)
            api_output = str(api_output)
            if "no tool founds" == api_output:
                break
            print("\033[32m" + response + "\033[0m" + "\033[34m" + ' ' + api_output + "\033[0m")
            prompt = prompt + response + ' ' + api_output
            response, _ = self.model.chat(self.tokenizer, prompt, history=None, stop_words_ids=react_stop_words_tokens)

        print("\033[32m" + response + "\033[0m")


# Example usage
qwen_integration = QwenLangChainIntegration()

query = "加拿大2022年的人口数量有多少？"
choose_tools = qwen_integration.tools
print("=" * 10)
qwen_integration.main(query, choose_tools)

query = "求解方程 2x+5 = -3x + 7"
choose_tools = qwen_integration.tools
print("=" * 10)
qwen_integration.main(query, choose_tools)

query = "编号是1605.08386的论文讲了些什么？"
choose_tools = qwen_integration.tools
print("=" * 10)
qwen_integration.main(query, choose_tools)

query = "使用python对下面的列表进行排序： [2, 4135, 523, 2, 3]"
choose_tools = qwen_integration.tools
print("=" * 10)
qwen_integration.main(query, choose_tools)
