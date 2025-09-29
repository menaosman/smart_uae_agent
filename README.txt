Smart Tourism Assistant for the UAE — LangChain Multi-Tool Agent
===================================================================

## What this is
A practical LangChain ReAct agent that:
- Answers UAE tourist questions from a local JSON knowledge base
- Provides real-time (or static) prayer times
- Estimates trip budgets (budget/standard/luxury)
- Generates LLM-based itineraries (grounded to JSON when possible)
- Remembers context with conversation memory

## Files
- smart_uae_agent.py — main script (run in console)
- uae_knowledge.json — starter knowledge base (edit/expand freely)
- SmartUAEAgent_Demo.ipynb — notebook version for step-by-step testing

## Requirements
Python 3.10+ recommended.
Install libs (choose the LLM you plan to use):
    pip install langchain langchain-core langchain-community requests
    pip install langchain-openai    # if using OpenAI (needs OPENAI_API_KEY)
    pip install langchain-google-genai  # if using Gemini (needs GEMINI_API_KEY)
    pip install langchain-groq      # if using Groq (needs GROQ_API_KEY)

## Environment
Set one of the following (depending on backend):
    export OPENAI_API_KEY=sk-...
    export GEMINI_API_KEY=...
    export GROQ_API_KEY=...

Optional for real prayer times:
    export USE_ALADHAN_API=true

Optional config:
    export SMARTUAE_LLM=openai|gemini|groq
    export SMARTUAE_KB=uae_knowledge.json

## Run (CLI)
    python smart_uae_agent.py --llm openai
Or rely on env var SMARTUAE_LLM.

Then try:
- "What are must-see attractions in Abu Dhabi?"
- "List top cultural tips for Sharjah"
- "Get prayer times: city=Dubai; date=2025-10-01"
- "Estimate my trip: city=Dubai; days=5; style=luxury"
- "Plan my 3-day itinerary in Ras Al Khaimah"

## Tools (Descriptions)
1) UAEKnowledgeSearchTool
   - Reads uae_knowledge.json and returns attractions, food, and cultural tips by city.
   - Example: "What’s the best thing to do in Ras Al Khaimah?"

2) PrayerTimeTool
   - Input: "city=<name>; date=YYYY-MM-DD"
   - Uses Aladhan API if USE_ALADHAN_API=true; otherwise returns static sample times (approximate).
   - Output: Fajr, Dhuhr, Asr, Maghrib, Isha.

3) TripBudgetPlanner
   - Inputs: "city; days; style"
   - Rates: budget=150 AED/day, standard=400 AED/day, luxury=1000 AED/day.

## LLM Tourist Recommendation
- The agent can directly produce itineraries (e.g., "Plan my 5-day trip to the UAE").
- It is encouraged (via system prompt) to ground choices in the JSON KB.
- You can expand uae_knowledge.json with more cities, events, and seasonal notes.

## Non-Functional Notes
- Agent aims to respond quickly (<5s) with max_iterations=4.
- Code is modular and commented; tools are reusable classes.
- ConversationBufferMemory preserves context across turns.

## Demo Video (How to record)
1. Run the script and record your terminal (e.g., with OBS, 720p, <2 minutes).
2. Show these interactions:
   - Ask about attractions in a city
   - Request a 5-day trip budget
   - Get prayer times for a city
   - Ask for a full itinerary
3. Save as demo.mp4.

## Jupyter Notebook
Open SmartUAEAgent_Demo.ipynb and run cells to test tool calls and the agent loop inside the notebook.
