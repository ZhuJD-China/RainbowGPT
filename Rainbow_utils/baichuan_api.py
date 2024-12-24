import requests
import json
from typing import List, Dict, Optional, Union, Any

class BaichuanAPI:
    """Baichuan API封装类"""
    
    def __init__(self, api_key: str):
        """
        初始化Baichuan API客户端
        
        Args:
            api_key (str): Baichuan API密钥
        """
        self.api_key = api_key
        self.api_base = "https://api.baichuan-ai.com/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "Baichuan3-Turbo-128k",
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_tokens: int = 4096,
        with_search_enhance: bool = True,
        knowledge_base_ids: Optional[List[str]] = None,
        stream: bool = True
    ) -> Union[str, Dict[str, Any]]:
        """
        调用Baichuan聊天完成API
        
        Args:
            messages (List[Dict[str, str]]): 消息列表，格式为[{"role": "user", "content": "消息内容"}]
            model (str): 模型名称
            temperature (float): 温度参数
            top_p (float): top_p参数
            max_tokens (int): 最大token数
            with_search_enhance (bool): 是否启用搜索增强
            knowledge_base_ids (List[str], optional): 知识库ID列表
            stream (bool): 是否使用流式输出
            
        Returns:
            Union[str, Dict[str, Any]]: API响应内容
        """
        try:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "with_search_enhance": with_search_enhance,
                "knowledge_base": {
                    "ids": knowledge_base_ids or []
                },
                "stream": stream
            }

            response = requests.post(
                self.api_base,
                headers=self.headers,
                json=payload
            )
            
            if response.status_code != 200:
                error_msg = f"API调用失败: HTTP {response.status_code}"
                try:
                    error_detail = response.json()
                    error_msg += f"\n详细信息: {json.dumps(error_detail, ensure_ascii=False)}"
                except:
                    error_msg += f"\n响应内容: {response.text}"
                raise Exception(error_msg)

            if stream:
                # 处理流式响应
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        try:
                            json_response = json.loads(line.decode('utf-8').replace('data: ', ''))
                            if 'choices' in json_response and len(json_response['choices']) > 0:
                                content = json_response['choices'][0].get('delta', {}).get('content', '')
                                if content:
                                    full_response += content
                        except json.JSONDecodeError:
                            continue
                return full_response
            else:
                # 处理非流式响应
                json_response = response.json()
                if 'choices' in json_response and len(json_response['choices']) > 0:
                    return json_response['choices'][0]['message']['content']
                return json_response

        except Exception as e:
            error_msg = f"Baichuan API调用出错: {str(e)}"
            print(error_msg)
            return error_msg 