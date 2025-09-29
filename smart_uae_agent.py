#!/usr/bin/env python3
"""
Smart Tourism Assistant for the UAE — LangChain Multi-Tool Agent
-----------------------------------------------------------------
Features:
- Knowledge search over local JSON (uae_knowledge.json)
- Prayer time lookup (static or Aladhan API)
- Trip budget estimation (budget/standard/luxury)
- LLM-powered trip recommendations (with grounding hints)
- ReAct-style tool-using agent with conversation memory
- Pluggable LLM backend: OpenAI / Gemini / Groq (choose via --llm or env)

Usage (CLI):
    python smart_uae_agent.py --llm openai
    python smart_uae_agent.py --llm gemini
    python smart_uae_agent.py --llm groq
Then type questions in the prompt.

Environment variables:
    OPENAI_API_KEY / GEMINI_API_KEY / GROQ_API_KEY
Optional:
    USE_ALADHAN_API=true  (to fetch real prayer times)
"""

import os
import json
import time
import argparse
import datetime as dt
from typing import Optional, Type, Any, Dict, List

# LangChain core
from langchain.tools import BaseTool
from langchain.agents import initialize_agent, AgentType, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage

# LLM backends (install the ones you need)
# pip install langchain-openai langchain-google-genai langchain-groq
LLM_BACKEND_ERR = None
ChatOpenAI = None
ChatGoogleGenerativeAI = None
ChatGroq = None

try:
    from langchain_openai import ChatOpenAI as _ChatOpenAI
    ChatOpenAI = _ChatOpenAI
except Exception as e:
    pass

try:
    from langchain_google_genai import ChatGoogleGenerativeAI as _ChatGoogleGenerativeAI
    ChatGoogleGenerativeAI = _ChatGoogleGenerativeAI
except Exception as e:
    pass

try:
    from langchain_groq import ChatGroq as _ChatGroq
    ChatGroq = _ChatGroq
except Exception as e:
    pass

import requests

# --------------------------
# Utility
# --------------------------
def load_knowledge(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_city(name: str) -> str:
    return (name or "").strip().lower()

# --------------------------
# Tool 1: UAEKnowledgeSearchTool
# --------------------------
class UAEKnowledgeSearchTool(BaseTool):
    name: str = "UAEKnowledgeSearchTool"
    description: str = (
        "Search the official local knowledge base about UAE cities, attractions, food, cultural tips, and general facts. "
        "Use this for factual questions like 'best things to do in Ras Al Khaimah' or 'cultural tips for Sharjah'."
    )

    def __init__(self, kb: Dict[str, Any]):
        super().__init__()
        self.kb = kb

    def _run(self, query: str) -> str:
        q = (query or "").lower()
        cities = self.kb.get("cities", {})
        hits: List[str] = []

        # try to find a city in the query
        city_hit = None
        for city in cities.keys():
            if city in q:
                city_hit = city
                break

        if city_hit:
            city_data = cities[city_hit]
            if "attractions" in q or "do in" in q or "things to do" in q or "what to do" in q:
                for a in city_data.get("attractions", []):
                    hits.append(f"- {a['name']}: {a['desc']}")
            if "food" in q or "restaurants" in q or "eat" in q:
                for r in city_data.get("food", []):
                    hits.append(f"- {r}")
            if "culture" in q or "cultural tips" in q or "dress" in q or "etiquette" in q:
                for tip in city_data.get("cultural_tips", []):
                    hits.append(f"- {tip}")
            # if no specific facet, return all
            if not hits:
                hits.append(f"Top in {city_hit.title()}:")
                for a in city_data.get("attractions", []):
                    hits.append(f"- {a['name']}: {a['desc']}")
        else:
            # generic country facts or cultural tips
            if "cultural" in q or "etiquette" in q or "dress" in q:
                for tip in self.kb.get("cultural_tips_general", []):
                    hits.append(f"- {tip}")
            if "fact" in q or "currency" in q or "weekend" in q or "transport" in q:
                for f in self.kb.get("country_facts", []):
                    hits.append(f"- {f}")

        if not hits:
            return "No exact match in knowledge base. Try specifying a UAE city like Dubai, Abu Dhabi, Sharjah, Ras Al Khaimah, or Fujairah."
        return "\n".join(hits)

    async def _arun(self, query: str) -> str:
        return self._run(query)

# --------------------------
# Tool 2: PrayerTimeTool
# --------------------------
class PrayerTimeTool(BaseTool):
    name: str = "PrayerTimeTool"
    description: str = (
        "Get Islamic prayer times for a UAE city and date. "
        "Input format: 'city=<city name>; date=YYYY-MM-DD'. "
        "Outputs Fajr, Dhuhr, Asr, Maghrib, Isha. Uses Aladhan API if USE_ALADHAN_API=true; otherwise a static fallback."
    )

    STATIC_TIMES = {
        # Simple static sample (approximate, not accurate). Use API for real results.
        "dubai": {"Fajr": "05:05", "Dhuhr": "12:18", "Asr": "15:42", "Maghrib": "18:07", "Isha": "19:28"},
        "abu dhabi": {"Fajr": "05:09", "Dhuhr": "12:22", "Asr": "15:46", "Maghrib": "18:10", "Isha": "19:31"},
        "sharjah": {"Fajr": "05:06", "Dhuhr": "12:19", "Asr": "15:43", "Maghrib": "18:08", "Isha": "19:29"},
        "ras al khaimah": {"Fajr": "05:03", "Dhuhr": "12:16", "Asr": "15:39", "Maghrib": "18:05", "Isha": "19:26"},
        "fujairah": {"Fajr": "05:01", "Dhuhr": "12:14", "Asr": "15:37", "Maghrib": "18:03", "Isha": "19:24"},
    }

    def _parse_input(self, s: str) -> Dict[str, str]:
        parts = [p.strip() for p in (s or "").split(";")]
        out = {}
        for p in parts:
            if "=" in p:
                k, v = p.split("=", 1)
                out[k.strip().lower()] = v.strip()
        return out

    def _fetch_from_aladhan(self, city: str, date: str) -> Optional[Dict[str, str]]:
        try:
            # Aladhan API docs: https://aladhan.com/prayer-times-api
            # We use the "By City" endpoint + date.
            d = dt.datetime.fromisoformat(date)
            params = {
                "city": city.title(),
                "country": "United Arab Emirates",
                "method": 2,
                "month": d.month,
                "year": d.year
            }
            url = "https://api.aladhan.com/v1/calendarByCity"
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            # find matching day
            for day in data.get("data", []):
                if str(day["date"]["gregorian"]["day"]).zfill(2) == str(d.day).zfill(2):
                    t = day["timings"]
                    # Remove extra timezone info like "+04:00"
                    clean = {k: v.split(" ")[0] for k, v in t.items() if k in ["Fajr", "Dhuhr", "Asr", "Maghrib", "Isha"]}
                    return clean
        except Exception as e:
            return None
        return None

    def _run(self, s: str) -> str:
        args = self._parse_input(s)
        city = normalize_city(args.get("city", ""))
        date = args.get("date") or dt.date.today().isoformat()

        use_api = os.getenv("USE_ALADHAN_API", "false").lower() == "true"
        times = None
        if use_api and city:
            times = self._fetch_from_aladhan(city, date)

        if times is None:
            times = self.STATIC_TIMES.get(city)

        if not times:
            return f"No times found for '{city}'. Try enabling USE_ALADHAN_API=true or choose a known UAE city."
        lines = [f"Prayer times for {city.title()} on {date}:"]
        for k in ["Fajr", "Dhuhr", "Asr", "Maghrib", "Isha"]:
            lines.append(f"- {k}: {times.get(k, 'N/A')}")
        return "\n".join(lines)

    async def _arun(self, s: str) -> str:
        return self._run(s)

# --------------------------
# Tool 3: TripBudgetPlanner
# --------------------------
class TripBudgetPlanner(BaseTool):
    name: str = "TripBudgetPlanner"
    description: str = (
        "Estimate total trip cost. Input: 'city=<city>; days=<int>; style=<budget|standard|luxury>'. "
        "Rates: budget=150 AED/day, standard=400 AED/day, luxury=1000 AED/day."
    )

    RATES = {"budget": 150, "standard": 400, "luxury": 1000}

    def _parse(self, s: str) -> Dict[str, str]:
        parts = [p.strip() for p in (s or "").split(";")]
        out = {}
        for p in parts:
            if "=" in p:
                k, v = p.split("=", 1)
                out[k.strip().lower()] = v.strip()
        return out

    def _run(self, s: str) -> str:
        args = self._parse(s)
        city = args.get("city", "UAE")
        days = int(args.get("days", "1"))
        style = args.get("style", "standard").lower()
        rate = self.RATES.get(style, self.RATES["standard"])
        total = rate * max(days, 1)
        return f"Estimated cost for {days} day(s) in {city.title()} ({style}): ~{total} AED (rate {rate} AED/day)."

    async def _arun(self, s: str) -> str:
        return self._run(s)

# --------------------------
# LLM factory
# --------------------------
def make_llm(backend: str, temperature: float = 0.3):
    backend = (backend or "openai").lower()
    if backend == "openai":
        if ChatOpenAI is None:
            raise RuntimeError("OpenAI backend not installed. pip install langchain-openai")
        return ChatOpenAI(model="gpt-4o-mini", temperature=temperature)
    elif backend == "gemini":
        if ChatGoogleGenerativeAI is None:
            raise RuntimeError("Gemini backend not installed. pip install langchain-google-genai")
        return ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=temperature)
    elif backend == "groq":
        if ChatGroq is None:
            raise RuntimeError("Groq backend not installed. pip install langchain-groq")
        return ChatGroq(model="llama-3.1-70b-versatile", temperature=temperature)
    else:
        raise ValueError("Unknown backend. Choose from: openai | gemini | groq")

# --------------------------
# System prompt (policy + behavior)
# --------------------------
SYSTEM_PROMPT = """You are SmartUAE, a helpful tourism assistant for the United Arab Emirates.
TOOLS POLICY:
- Use tools for FACTS (knowledge JSON, prayer times, budgets). Do NOT invent facts.
- For trip ideas/itineraries: you may answer directly using your general knowledge, but prefer grounding to the JSON knowledge when possible.
- If you are not sure, ask for a city or trip duration, or say you don't have that info.
CONTEXT POLICY:
- Remember city, dates, and trip style from earlier turns (ConversationBufferMemory).
STYLE:
- Be concise, friendly, and structured. Use bullet points and day-by-day lists for plans.
- Always include AED for any money values.
"""

# --------------------------
# Agent setup
# --------------------------
def build_agent(knowledge_path: str, llm_backend: str = "openai") -> AgentExecutor:
    kb = load_knowledge(knowledge_path)

    tools: List[BaseTool] = [
        UAEKnowledgeSearchTool(kb),
        PrayerTimeTool(),
        TripBudgetPlanner(),
    ]

    llm = make_llm(llm_backend, temperature=0.3)

    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )

    # We add a system message to nudge the agent
    system_msg = SystemMessage(content=SYSTEM_PROMPT)

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=4,
        early_stopping_method="generate",
        agent_kwargs={"extra_prompt_messages": [system_msg]},
    )

    return agent

# --------------------------
# CLI loop
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", default=os.getenv("SMARTUAE_LLM", "openai"), help="openai|gemini|groq")
    parser.add_argument("--kb", default=os.getenv("SMARTUAE_KB", "uae_knowledge.json"))
    args = parser.parse_args()

    agent = build_agent(args.kb, args.llm)

    print("SmartUAE Agent ready. Ask me about UAE tourism, budgets, or prayer times.")
    print("Examples:")
    print("  - What are must-see attractions in Abu Dhabi?")
    print("  - Get prayer times: city=Dubai; date=2025-10-01")
    print("  - Estimate my trip: city=Dubai; days=5; style=luxury")
    print("  - Plan me a 3-day trip in Ras Al Khaimah.")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if q.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        t0 = time.time()
        try:
            resp = agent.invoke({"input": q})
            out = resp.get("output") or resp
        except Exception as e:
            out = f"[Error] {e}"
        dt_ms = int((time.time() - t0) * 1000)
        print(f"\nSmartUAE ({dt_ms} ms): {out}\n")

if __name__ == "__main__":
    main()
