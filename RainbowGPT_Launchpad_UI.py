import gradio as gr
import RainbowChromadb_Option
import RainbowKnowledge_Agent
import RainbowSQL_Agent
import RainbowStock_Analysis
from Rainbow_utils.get_gradio_theme import Seafoam

seafoam = Seafoam()

RainbowKnowledge_Agent = RainbowKnowledge_Agent.RainbowKnowledge_Agent().launch()
RainbowSQL_Agent = RainbowSQL_Agent.RainbowSQLAgent().launch()
RainbowStock_Analysis = RainbowStock_Analysis.RainbowStock_Analysis().launch()
ChromaDBGradioUI = RainbowChromadb_Option.ChromaDBGradioUI().launch()

RainbowGPT_TabbedInterface = gr.TabbedInterface(
    [RainbowKnowledge_Agent, RainbowSQL_Agent, RainbowStock_Analysis, ChromaDBGradioUI],
    ["Rainbow-Knowledge-Agent", "Rainbow-SQL-Agent", "Rainbow-Stock-Analysis"
        , "ChromaDB-Option"]
    , theme=seafoam)

if __name__ == "__main__":
    RainbowGPT_TabbedInterface.queue().launch()
