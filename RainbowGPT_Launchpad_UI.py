import gradio as gr
import RainbowKnowledge_Agent
import RainbowSQL_Agent
import RainbowStock_Analysis
import RainbowChromadb_Option
from Rainbow_utils.get_gradio_theme import Seafoam
from Rainbow_utils.set_csv_2_MySQL_uploader import CSVToMySQLUploader

seafoam = Seafoam()

RainbowKnowledge_Agent = RainbowKnowledge_Agent.RainbowKnowledge_Agent().launch()
RainbowSQL_Agent = RainbowSQL_Agent.RainbowSQLAgent().launch()
RainbowStock_Analysis = RainbowStock_Analysis.RainbowStock_Analysis().launch()
ChromaDBGradioUI = RainbowChromadb_Option.ChromaDBGradioUI().launch()
CSVToMySQLUploader = CSVToMySQLUploader().launch()

RainbowGPT_TabbedInterface = gr.TabbedInterface(
    [RainbowKnowledge_Agent, RainbowSQL_Agent,
     RainbowStock_Analysis, ChromaDBGradioUI, CSVToMySQLUploader],
    ["Rainbow-Knowledge-Agent", "Rainbow-SQL-Agent", "Rainbow-Stock-Analysis"
        , "ChromaDB-Option", "CSV-2-MySQL-Uploader"]
    , theme=seafoam)

if __name__ == "__main__":
    RainbowGPT_TabbedInterface.queue().launch()
