import logging

from dotenv import load_dotenv
_ = load_dotenv(override=True)

logger = logging.getLogger("dlai-agent")
logger.setLevel(logging.INFO)

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli #, voice_assistant
from livekit.agents.llm import ChatContext, ChatMessage
from livekit.plugins import (
    openai,
    silero,
    deepgram,
    google
)
from livekit.agents import room_io
import datetime

from vrag import VRAG


# deeogram.secret = "d7135beef77cff4f58fc5f264ea9f3b16f69b2cb"
# deepgram.project_id = "voicepoc1"



class Assistant(Agent):
    def __init__(self, vector_store=None) -> None:
        self.vector_store = vector_store
        
        llm = openai.LLM.with_ollama(model="deepseek-r1:1.5b", base_url="http://localhost:11434/v1")
        # llm = openai.LLM.with_ollama(model="gemma3:12b", base_url="http://localhost:11434/v1")
        # llm = google.LLM(model="gemini/gemini-2.5-flash", temperature=0.7)
        # llm = google.LLM(model="gemini-2.5-flash-lite", temperature=0.7) #gemini-3-flash-preview
        # llm = google.LLM(model="gemini-3-flash-preview", temperature=0.7) #
        stt = deepgram.STT(model="nova-2",language="en")
        
        tts = deepgram.TTS(model="aura-asteria-en")

        # tts = openai.TTS.(
        #         base_url="http://localhost:11434/v1", # Ollama's local endpoint
        #         api_key="ollama",
        #         model="sematre/orpheus:ft-hi-3b-q8_0",
        #         # model="legraphista/Orpheus:3b-ft-q4_k_m" 
        #     )

        silero_vad = silero.VAD.load()

        super().__init__(
            instructions="""
                You are a helpful assistant communicating 
                via voice. Use the provided context to answer questions accurately.
            """,
            stt=stt,
            llm=llm,
            tts=tts,
            vad=silero_vad,
        )
    
    #def _on_user_turn_completed(self, turn):
    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage,) -> None:
        """Retrieve relevant context from VRAG and add it to the chat history"""
        print("ChatContext:")
        for i, item in enumerate(turn_ctx.items):
            print(f"  {i}: {item}")
        print("New Message:", new_message)
        #turn = turn_ctx
        if self.vector_store is None:
            print("No vector store available, skipping context retrieval")
            return
        
        # Get the user's message from the turn
        # user_message = None
        # for msg in turn_ctx.items:
        #     if msg.role == "user":
        #         user_message = msg.content
        #         break

        user_message = new_message.content[-1] if new_message.role == "user" else None
        

        if not user_message:
            print("No user message found in turn, skipping context retrieval")
            return
        
        # Retrieve relevant context from the vector store
        try:
            docs = self.vector_store.similarity_search(user_message, k=3)
            context = "\n".join([doc.page_content for doc in docs])
            #print("Context", context)
            
            # Add context as a system message to enrich the LLM's response
            if context:
                context_message = f"Context information:\n{context}"
                # print(context_message)

                # Insert context before the user's message in the chat history
                turn_ctx.add_message(
                    role="assistant", # or "user", "system", etc.
                    content=[context_message]
                )
                # turn_ctx.items.insert(
                #     len(turn_ctx.items) - 1,
                #     ChatMessage.system(context_message)
                # )
                logger.info(f"Added {len(docs)} context documents to user turn")
        except Exception as e:
            logger.error(f"Error retrieving context from VRAG: {e}")




async def entrypoint(ctx: JobContext):
    # Initialize VRAG with a document
    vrag_instance = VRAG()
    # Load your document here - adjust the path as needed
    # vector_store = vrag_instance.get_vector_store(r"rag_docs/ResumeShazeb.pdf")
    vector_store = vrag_instance.get_vector_store_from_folder("rag_docs")
    
    await ctx.connect()

    session = AgentSession()

    await session.start(
        room=ctx.room,
        agent=Assistant(vector_store=vector_store)
    )


    # Give a greeting
    # await session.generate_reply("Hello, I am running DeepSeek locally. How can I help you?") # , allow_interruptions=True)
    await session.say("Hello") # , allow_interruptions=True)

    @session.on("user_input_transcribed")
    def on_transcript(transcript):
        if transcript.is_final:
            print(f"[{datetime.datetime.now()}] {transcript.transcript}")



if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))

# jupyter.run_app(
#     WorkerOptions(entrypoint_fnc=entrypoint), 
#     jupyter_url="https://jupyter-api-livekit.vercel.app/api/join-token"
# )

# Configure Logging
# logger = logging.getLogger("voice-agent")
# logger.setLevel(logging.INFO)

# async def entrypoint(ctx: JobContext):
#     # 1. Define the LLM (DeepSeek R1 via Ollama)
#     # Ensure Ollama is running on localhost:11434
#     llm = ollama.LLM(model="deepseek-r1:1.5b")

#     # 2. Define the STT (Silero is lightweight and runs locally)
#     stt = silero.STT()

#     # 3. Define the TTS (Using a basic local engine)
#     # Note: For a "basic" local setup, we use Silero or a simple local provider
#     tts = silero.TTS() 

#     # 4. Build the Pipeline Agent
#     agent = voice_assistant.VoiceAssistant(
#         vad=silero.VAD.load(), # Voice Activity Detection
#         stt=stt,
#         llm=llm,
#         tts=tts,
#     )

#     # Connect to the room and start the agent
#     await ctx.connect()
#     agent.start(ctx.room)

#     # Give a greeting
#     await agent.say("Hello, I am running DeepSeek locally. How can I help you?", allow_interruptions=True)


# if __name__ == "__main__":
#     cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))