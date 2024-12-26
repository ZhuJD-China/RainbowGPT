# 公共前部分
common_text_before = """
嗨！让我来帮你找答案。我会：

- 仔细理解你的问题
- 用最合适的关键词搜索
- 避免重复查找已有的内容
- 最多搜索两次，确保效率

记住，我的目标是给你一个清晰、准确、有用的答案 😊
"""

# 公共后部分
common_text_after = """
关于你的问题 "{human_input_first}"
当前搜索词: "{human_input}"

让我看看:
- 现有信息是否够用了
- 是否需要继续找（但不重复之前找过的）
- 怎么给你最有价值的回答
"""

local_search_template = common_text_before + """
我在知识库里找到了这些:
"{combined_text}"

让我整理一下这些信息...
""" + common_text_after

google_search_template = common_text_before + """
我在网上找到了这些:
"{combined_text}"

让我帮你梳理一下这些内容...
""" + common_text_after
