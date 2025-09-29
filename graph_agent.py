import os, json, datetime as dt, requests
from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation, create_react_agent
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI  # or use Gemini/Groq analogs

# ---- Tools (simple wrappers around your existing ones) ----
from smart_uae_agent import UAEKnowledgeSearchTool, PrayerTimeTool, TripBudgetPlanner

class GraphState(BaseModel):
    messages: List[dict] = Field(default_factory=list)

def build_graph_agent(kb_path: str, llm_backend: str = "openai"):
    # Choose LLM
    if llm_backend == "openai":
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    else:
        raise ValueError("This starter uses OpenAI; add others as needed.")

    tools = [
        UAEKnowledgeSearchTool(kb_path=kb_path),
        PrayerTimeTool(),
        TripBudgetPlanner(),
    ]

    # Prebuilt ReAct-style agent with tool-calling
    agent = create_react_agent(llm, tools)

    # Build graph with a single agent node (you can add memory/persistence later)
    def call_agent(state: GraphState):
        result = agent.invoke({"messages": state.messages})
        # Append result to messages
        return {"messages": state.messages + [{"role": "assistant", "content": result["messages"][-1].content}]}

    graph = StateGraph(GraphState)
    graph.add_node("agent", call_agent)
    graph.set_entry_point("agent")
    graph.add_edge("agent", END)
    return graph.compile()

if __name__ == "__main__":
    kb = os.getenv("SMARTUAE_KB", "uae_knowledge.json")
    app = build_graph_agent(kb)
    # quick demo:
    out = app.invoke({"messages": [{"role":"user","content":"What are must-see attractions in Abu Dhabi?"}]})
    print(out["messages"][-1]["content"])
