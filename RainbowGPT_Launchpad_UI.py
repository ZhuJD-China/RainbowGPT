import gradio as gr
import RainbowKnowledge_Agent
import RainbowSQL_Agent
import RainbowStock_Analysis
import RainbowChromadb_Option
from Rainbow_utils.get_gradio_theme import Seafoam
from Rainbow_utils.set_csv_2_MySQL_uploader import CSVToMySQLUploader
from RainbowModel_Manager import RainbowModelManager

seafoam = Seafoam()

# 创建模型管理器界面
RainbowModel_Manager = RainbowModelManager().launch()
RainbowKnowledge_Agent = RainbowKnowledge_Agent.RainbowKnowledge_Agent().launch()
RainbowSQL_Agent = RainbowSQL_Agent.RainbowSQLAgent().launch()
RainbowStock_Analysis = RainbowStock_Analysis.RainbowStock_Analysis().launch()
ChromaDBGradioUI = RainbowChromadb_Option.ChromaDBGradioUI().launch()
CSVToMySQLUploader = CSVToMySQLUploader().launch()

RainbowGPT_TabbedInterface = gr.TabbedInterface(
    [RainbowModel_Manager, RainbowKnowledge_Agent, ChromaDBGradioUI,
     RainbowSQL_Agent, CSVToMySQLUploader,
     RainbowStock_Analysis],
    ["Model-Config", "Rainbow-Knowledge-Agent", "Knowledge-ChromaDB-Option",
     "Rainbow-SQL-Agent", "CSV-2-MySQL-Uploader",
     "Rainbow-Stock-Analysis"]
    , theme=seafoam)

if __name__ == "__main__":
    RainbowGPT_TabbedInterface.queue().launch(share=True)
