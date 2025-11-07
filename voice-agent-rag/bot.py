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
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair

from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.google.stt import GoogleSTTService
from pipecat.transcriptions.language import Language

from pipecat.services.google.llm import GoogleLLMService
# from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService

from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService

from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams

from customer_support import SupportCrew

from deepgram import (
    LiveOptions
)


# from pipecat_whisker import WhiskerObserver

load_dotenv(override=True)

support_team = SupportCrew()

support_team.crew()

# Initialize the client globally
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

SYSTEM_INSTRUCTION = f"""
Du bist Laura, eine freundliche und hilfsbereite Mitarbeiterin.
Deine Aufgabe ist es Dein Wissen auf prägnante Art und Weise mitzuteilen.
Formuliere alle Antworten in deutscher Sprache.
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

async def query_knowledge_base(params: FunctionCallParams):
    """Query the knowledge base for the answer to the question."""
    logger.info(f"Querying knowledge base for question: {params.arguments['question']}")
    logger.info(params.context.get_messages())

    # for our case, the first two messages are the instructions and the user message
    # so we remove them.
    conversation_turns = params.context.get_messages()[2:]

    def _get_field(turn, name, default=None):
        if isinstance(turn, dict):
            return turn.get(name, default)
        # Pydantic v2 models
        if hasattr(turn, "model_dump"):
            data = turn.model_dump()
            return data.get(name, default)
        return getattr(turn, name, default)

    def _is_tool_call(turn):
        if _get_field(turn, "role") == "tool":
            return True
        tool_calls = _get_field(turn, "tool_calls")
        return bool(tool_calls)

    # filter out tool calls
    messages = [turn for turn in conversation_turns if not _is_tool_call(turn)]
    logger.info(f"Messages: {messages}")

    # use the last 3 turns as the conversation history/context
    messages = messages[-3:]
    def _to_plain(turn):
        if isinstance(turn, dict):
            return turn
        if hasattr(turn, "model_dump"):
            return turn.model_dump()
        return {
            "role": getattr(turn, "role", None),
            "content": getattr(turn, "parts", getattr(turn, "content", None)),
        }

    messages_json = json.dumps([_to_plain(m) for m in messages], ensure_ascii=False, indent=2)

    logger.info(f"Conversation turns: {messages_json}")

    start = time.perf_counter()

    full_prompt = f"Conversation History: {messages_json}"
    response = SupportCrew().crew().kickoff(inputs={
        "customer": "Schaefer.AI",
        "person": "Siegfried Schaefer",
        "inquiry": f"{full_prompt}"
    })

    end = time.perf_counter()
    logger.info(f"Time taken: {end - start:.2f} seconds")
    logger.info(response)

    await params.result_callback({"answer": f"{response}"})


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
        description="query infos about a temperature controlunit .",
        properties={
            "question": {
                "type": "string",
                "description": "The question to query informations about a temperature control unit.",
            },
        },
        required=["question"],
    )

    tools = ToolsSchema(standard_tools=[get_datetime_function, query_function])

#    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"),
#                             live_options=LiveOptions (
#                                 language='de'
#                             ))

    stt = GoogleSTTService(
        params=GoogleSTTService.InputParams(languages=Language.DE_DE, model="chirp_3"),
        credentials=os.getenv("GOOGLE_TEST_CREDENTIALS"),
        location="us",
    )


#    tts = CartesiaTTSService(
#        api_key=os.getenv("CARTESIA_API_KEY"),
#        model_id="sonic-3",
#        voice_id="576a28db-95fe-4b40-adaa-69a4249e8085",  # Laura
#    )

    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY", ""),
        voice_id=os.getenv("ELEVENLABS_SIGGIS_ID", "")
    )


    llm = GoogleLLMService(api_key=os.getenv("GOOGLE_API_KEY"), 
                           model="gemini-2.0-flash-001",
                           language="En_EN",
                           tools=tools,
                           system_instruction=SYSTEM_INSTRUCTION)


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
        {
            "role": "user",
            "content": "Beginne damit, dich kurz vorzustellen. Formuliere alle Antworten in einem guten Deutsch und auch darauf Zahlen und Uhrzeiten auf Deutsch zu nennen."
        }
    ]

#    GeminiLIveApi    
#    context = OpenAILLMContext(messages, tools)
#    context_aggregator = llm.create_context_aggregator(context)

    context = LLMContext(messages, tools)
    context_aggregator = LLMContextAggregatorPair(context)

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

    # whisker = WhiskerObserver(pipeline)

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=300,
        # observers=[whisker]
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
    
