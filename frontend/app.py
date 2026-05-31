import asyncio
import chainlit as cl

# --- MOCK BACKEND ENGINE ---
# This fakes the stream you will eventually get from Person B's LangGraph
async def mock_deepseek_stream(user_message: str):
    # Fake reasoning trace (The CoT)
    thinking_tokens = [
        "<think>\n",
        "Analyzing user input...\n",
        f"Detected query: '{user_message}'\n",
        "Checking grammar rules in ChromaDB... Match found: [Spanish Present Tense].\n",
        "Pedagogical Strategy: The user made a common error. Do NOT provide the direct translation.\n",
        "Formulating a Socratic hint...\n",
        "</think>\n"
    ]
    
    # Fake final response (The Hint)
    response_tokens = [
        "¡Hola! Let's take a closer look at the verb you used in that sentence.\n\n",
        "You wrote a phrase using the subject **'Yo'** (I), but look closely at the verb ending you chose.\n\n",
        "For regular **-ar** verbs in the present tense, what is the correct ending when *you* are the one performing the action? Give it another try!"
    ]
    
    # Stream the <think> tokens slowly
    for token in thinking_tokens:
        await asyncio.sleep(0.15)
        yield token
        
    # Stream the final response normally
    for token in response_tokens:
        await asyncio.sleep(0.08)
        yield token


# --- CHAINLIT FRONTEND UI ---
@cl.on_chat_start
async def start():
    # The welcome message when a student opens the app
    await cl.Message(content="Welcome to **MrWorldwide** 🌍! I am your Socratic language coach. Write a sentence in your target language!").send()

@cl.on_message
async def main(message: cl.Message):
    # 1. Create a collapsible "Step" box for the Teacher's Logic
    teacher_logic_step = cl.Step(name="Teacher's Internal Logic", type="run")
    await teacher_logic_step.send()
    
    # 2. Create the placeholder for the final student-facing message
    student_response_message = cl.Message(content="")
    
    is_thinking_mode = False
    
    # 3. Stream from our mock backend function
    async for token in mock_deepseek_stream(message.content):
        
        # Intercept the start of the <think> block
        if "<think>" in token:
            is_thinking_mode = True
            teacher_logic_step.output = "" 
            continue
            
        # Intercept the end of the </think> block
        if "</think>" in token:
            is_thinking_mode = False
            await teacher_logic_step.update() # This visually collapses the box!
            continue
            
        # 4. Route the text to the correct UI element
        if is_thinking_mode:
            await teacher_logic_step.stream_token(token)
        else:
            await student_response_message.stream_token(token)
            
    # Send the final completed message to the chat
    await student_response_message.send()