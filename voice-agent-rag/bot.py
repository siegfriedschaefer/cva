#
# Copyright (c) 2025, Daily
# Copyright (c) 2025, Siegfried Schaefer
#
# SPDX-License-Identifier: BSD 2-Clause License
#
import os
import sys

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext

from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams

from google.genai import types

load_dotenv(override=True)

#SYSTEM_INSTRUCTION = f"""
#"You are Gemini Chatbot, a friendly, helpful robot.
#
#Your goal is to demonstrate your capabilities in a succinct way.
#
#Your output will be converted to audio so don't include special characters in your answers.
#
#Respond to what the user said in a creative and helpful way. Keep your responses brief. One or two sentences at most.
#"""

SYSTEM_INSTRUCTION = f"""
Du bist Laura, ein freundlicher und hilfsbereiter KI Agent.
Deine Aufgabe ist es Deine Fähigkeiten auf prägnante Art und Weise zu demonstrieren.
Deine Antworten werden in eine Audiodatei umgewandelt, verzichte bitte auf Sonderzeichen.
"""

async def get_current_datetime(params: FunctionCallParams):
    """Gets the current date and time."""
    from datetime import datetime
    datetimestr = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    await params.result_callback({"answer":f"Heute ist der {datetimestr}."})


async def run_bot(webrtc_connection):
    pipecat_transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            audio_out_10ms_chunks=2,
        ),
    )

    get_current_datetime_function = FunctionSchema(
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
    tools = ToolsSchema(standard_tools=[get_current_datetime_function])



    llm = GeminiLiveLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        voice_id="Aoede",  #  Aoede, Charon, Fenrir, Kore, Puck
        language="DE_DE",
        transcribe_user_audio=True,
        transcribe_model_audio=True,
        tools=tools
#        system_instruction=SYSTEM_INSTRUCTION,
    )
    llm.register_function("get_datetime", get_current_datetime)


    messages = [
        {
            "role": "system", 
            "content":
                        """\
                        Du bist Laura, ein freundlicher und hilfsbereiter KI Agent.
                        Deine Aufgabe ist es Deine Fähigkeiten auf prägnante Art und Weise zu demonstrieren.
                        Deine Antworten werden in eine Audiodatei umgewandelt, verzichte bitte auf Sonderzeichen.
                        Du kannst auf Fragen nach dem Datum oder der Uhrzeit mit der Funktion get_datetime reagieren.

                        You have access to the following tools: get_datetime.

                        You can respond to questions about the time and date using the get_datetime tool.

                        """
        },
        {
            "role": "user",
#                "content": "Start by greeting the user warmly and introducing yourself.",
            "content": "Beginne damit, den Benutzer herzlich zu begrüßen und dich vorzustellen."
        }
    ]
    context = OpenAILLMContext(messages, tools)
    context_aggregator = llm.create_context_aggregator(context)
#    context_aggregator = LLMContextAggregatorPair(context)


# excerpt from https://www.cerebrium.ai/blog/creating-a-realtime-rag-voice-agent

#    chain = prompt | ChatOpenAI(model="gpt-4o", temperature=0.7)
#        history_chain = RunnableWithMessageHistory(
#            chain,
#            get_session_history,
#            history_messages_key="chat_history",
#            input_messages_key="input")
#        lc = LangchainProcessor(history_chain)

    pipeline = Pipeline(
        [
            pipecat_transport.input(),
            context_aggregator.user(),
            llm,  # LLM
            pipecat_transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
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
    