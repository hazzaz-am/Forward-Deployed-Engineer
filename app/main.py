import asyncio
from dotenv import load_dotenv
from fastapi import Body, FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from langchain_openai import ChatOpenAI

load_dotenv()

SYSTEM_PROMPT = """\
You are a customer-support executive for our
Food ordering app named Tomato.

Your job is to identify the customer's main
problem and urgency. Answer them related to there query.

Use professional language. If user has an issue,
use words like I understand your frustration,
I am really sorry for your trouble etc.

Do not answer any other question which is not
related to Ordering Food query, refund query,
order tracking status query or company policy query.
"""



app = FastAPI(title="Tomato Chat API")
app.add_middleware(
  CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

history: list[dict[str, str]] = []
history_lock = asyncio.Lock()

llm = ChatOpenAI(model="gpt-4o-mini",
                 use_responses_api=True,
                 store=False,
                 model_kwargs={"instructions": SYSTEM_PROMPT}
                )

@app.post("/api/chat", response_class=PlainTextResponse)
async def chat(message: str = Body(..., media_type="text/plain")) -> str:
  async with history_lock:
    history.append({"role": "user", "content": message})

    api_response = await llm.ainvoke(history)

    history.append({"role": "assistant", "content": api_response.text})
    return api_response.text