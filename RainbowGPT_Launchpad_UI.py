import gradio as gr
import RainbowKnowledge_Agent
import RainbowSQL_Agent
import RainbowSQL_Agent_Custom
import RainbowStock_Analysis
import RainbowChromadb_Option
from Rainbow_utils.get_gradio_theme import Seafoam
from Rainbow_utils.set_csv_2_MySQL_uploader import CSVToMySQLUploader
from RainbowModel_Manager import RainbowModelManager

seafoam = Seafoam()

# 创建各个界面实例
RainbowModel_Manager = RainbowModelManager().launch()
RainbowKnowledge_Agent_UI = RainbowKnowledge_Agent.RainbowKnowledge_Agent().launch()
RainbowSQL_Agent_UI = RainbowSQL_Agent.RainbowSQLAgent().launch()
RainbowSQL_Agent_Custom_UI = RainbowSQL_Agent_Custom.RainbowSQLAgentCustom().launch()
RainbowStock_Analysis_UI = RainbowStock_Analysis.RainbowStock_Analysis().launch()
ChromaDBGradioUI = RainbowChromadb_Option.ChromaDBGradioUI().launch()
CSVToMySQLUploader_UI = CSVToMySQLUploader().launch()

# 创建知识库相关的标签页组
knowledge_tabs = gr.TabbedInterface(
    [RainbowKnowledge_Agent_UI, ChromaDBGradioUI],
    ["Knowledge Agent", "ChromaDB Options"]
)

# 创建SQL相关的标签页组
sql_tabs = gr.TabbedInterface(
    [RainbowSQL_Agent_UI, RainbowSQL_Agent_Custom_UI, CSVToMySQLUploader_UI],
    ["SQL Agent Custom", "CSV to MySQL Uploader"]
)

# 创建主界面
RainbowGPT_TabbedInterface = gr.TabbedInterface(
    [RainbowModel_Manager, knowledge_tabs, sql_tabs, RainbowStock_Analysis_UI],
    ["Model Config", "Knowledge Agent", "SQL Agent", "Stock Analysis"],
    theme=seafoam
)

if __name__ == "__main__":
    RainbowGPT_TabbedInterface.queue().launch(share=True)
