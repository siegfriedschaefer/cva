#
# Copyright (c) 2025, Daily
# Copyright (c) 2025, Siegfried Schaefer
#
# SPDX-License-Identifier: BSD 2-Clause License
#
import os
import time
import json
import asyncio


from dotenv import load_dotenv
from loguru import logger

from google import genai

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair


from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService

from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService

from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams

# from pipecat_whisker import WhiskerObserver

load_dotenv(override=True)

# Initialize the client globally
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])


def get_rag_content():
    """Get the RAG content from the file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rag_content_path = os.path.join(script_dir, "assets", "rag-content.txt")
    with open(rag_content_path, "r") as f:
        return f.read()



SYSTEM_INSTRUCTION = f"""
Du bist Laura, ein freundlicher und hilfsbereiter KI Agent.
Deine Aufgabe ist es Deine Fähigkeiten auf prägnante Art und Weise zu demonstrieren.
Bitte formuliere alle Antworten in deutscher Sprache.
Deine Antworten werden in eine Audiodatei umgewandelt, verzichte bitte auf Sonderzeichen.
"""

async def get_datetime(params: FunctionCallParams):
    """Get the current date and time as a structured response."""
    from datetime import datetime
    datestr = datetime.now().strftime("%Y-%m-%d")
    clockstr = datetime.now().strftime("%H:%M:%S")
    await params.result_callback({
        "datum": f"Heute ist der {datestr}.",
        "uhrzeit": f"Es ist {clockstr} Uhr."
    })

RAG_MODEL = "gemini-2.0-flash-lite-preview-02-05"
VOICE_MODEL = "gemini-2.0-flash"
RAG_CONTENT = get_rag_content()
RAG_PROMPT = f"""
You are a helpful assistant designed to answer user questions based solely on the provided knowledge base.

**Instructions:**

1.  **Knowledge Base Only:** Answer questions *exclusively* using the information in the "Knowledge Base" section below. Do not use any outside information.
2.  **Conversation History:** Use the "Conversation History" (ordered oldest to newest) to understand the context of the current question.
3.  **Concise Response:**  Respond in 50 words or fewer.  The response will be spoken, so avoid symbols, abbreviations, or complex formatting. Use plain, natural language.
4.  **Unknown Answer:** If the answer is not found within the "Knowledge Base," respond with "I don't know." Do not guess or make up an answer.
5. Do not introduce your response. Just provide the answer.
6. You must follow all instructions.

**Input Format:**

Each request will include:

*   **Conversation History:**  (A list of previous user and assistant messages, if any)

**Knowledge Base:**
Here is the knowledge base you have access to:
{RAG_CONTENT}
"""

async def query_knowledge_base(params: FunctionCallParams):
    """Query the knowledge base for the answer to the question."""
    logger.info(f"Querying knowledge base for question: {params.arguments['question']}")
    logger.info(params.context.get_messages())

    # for our case, the first two messages are the instructions and the user message
    # so we remove them.
    conversation_turns = params.context.get_messages()[2:]

    def _is_tool_call(turn):
        if turn.get("role", None) == "tool":
            return True
        if turn.get("tool_calls", None):
            return True
        return False

    # filter out tool calls
    messages = [turn for turn in conversation_turns if not _is_tool_call(turn)]
    logger.info(f"Messages: {messages}")

    # use the last 3 turns as the conversation history/context
    messages = messages[-3:]
    messages_json = json.dumps(messages, ensure_ascii=False, indent=2)

    logger.info(f"Conversation turns: {messages_json}")

    start = time.perf_counter()
    full_prompt = f"System: {RAG_PROMPT}\n\nConversation History: {messages_json}"

    response = await client.aio.models.generate_content(
        model=RAG_MODEL,
        contents=[full_prompt],
        config={
            "temperature": 0.1,
            "max_output_tokens": 64,
        },
    )

    end = time.perf_counter()
    logger.info(f"Time taken: {end - start:.2f} seconds")
    logger.info(response.text)

    await params.result_callback({"answer": f"{response.text}"})

async def run_bot(webrtc_connection):
    pipecat_transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            audio_out_10ms_chunks=4,
        ),
    )

    get_datetime_function = FunctionSchema(
        name="get_datetime",
        description="Get current date and time.",
        properties={
            "question": {
                "type": "string",
                "description": "The question that the user is asking about the current date and time.",
            }
        },
        required=["question"],
    )

    query_function = FunctionSchema(
        name="query_knowledge_base",
        description="query infos about family Fischer.",
        properties={
            "question": {
                "type": "string",
                "description": "The question to query informations about family Fischer.",
            },
        },
        required=["question"],
    )

    tools = ToolsSchema(standard_tools=[get_datetime_function, query_function])

    llm = GoogleLLMService(api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-2.0-flash-001")


#    llm = GeminiLiveLLMService(
#        api_key=os.getenv("GOOGLE_API_KEY"),
#        voice_id="Aoede",  #  Aoede, Charon, Fenrir, Kore, Puck
#        language="DE_DE",
#        transcribe_user_audio=True,
#        transcribe_model_audio=True,
#        tools=tools,
#        system_instruction=SYSTEM_INSTRUCTION,
#    )
 
 
    llm.register_function("get_datetime", get_datetime)
    llm.register_function("query_knowledge_base", query_knowledge_base)

    messages = [
#        {
#            "role": "system", 
#            "content":
#                        """\
#                        Du bist Laura, ein freundlicher und hilfsbereiter KI Agent.
#                        Deine Aufgabe ist es Deine Fähigkeiten auf prägnante Art und Weise zu demonstrieren.
#                        Deine Antworten werden in eine Audiodatei umgewandelt, verzichte bitte auf Sonderzeichen.
#                        Du kannst auf Fragen nach dem Datum oder der Uhrzeit mit der Funktion get_datetime reagieren.##
#
#                        Folgende Tools stehen Dir zur Verfuegung: get_datetime.#
#
#                        Du kannst auf Fragen nach dem Datum oder der Uhrzeit mit der Funktion get_datetime beantworten.
#
#                        """
#        },
        {
            "role": "user",
#                "content": "Start by greeting the user warmly and introducing yourself.",
            "content": "Beginne damit, den Benutzer herzlich zu begrüßen und dich vorzustellen."
        }
    ]
    context = OpenAILLMContext(messages, tools)
    context_aggregator = llm.create_context_aggregator(context)

# excerpt from https://www.cerebrium.ai/blog/creating-a-realtime-rag-voice-agent

#    chain = prompt | ChatOpenAI(model="gpt-4o", temperature=0.7)
#        history_chain = RunnableWithMessageHistory(
#            chain,
#            get_session_history,
#            history_messages_key="chat_history",
#            input_messages_key="input")
#        lc = LangchainProcessor(history_chain)





    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )

    pipeline = Pipeline(
        [
            pipecat_transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            pipecat_transport.output(),
            context_aggregator.assistant(),
        ]
    )





    pipeline = Pipeline(
        [
            pipecat_transport.input(),
            context_aggregator.user(),
            llm,
            pipecat_transport.output(),
            context_aggregator.assistant(),
        ]
    )

#    whisker = WhiskerObserver(pipeline)


    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=10,
#        observers=[whisker]
#        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @pipecat_transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Pipecat Client connected")
        # Kick off the conversation.
        await task.queue_frames([LLMRunFrame()])

    @pipecat_transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Pipecat Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)

    await runner.run(task)
    
