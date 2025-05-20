from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, MessagesState
from murf import Murf
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AgentState(MessagesState):
    """State holding the raw text, its summary, and the generated script."""
    text: str = ""
    summary: str = ""
    script: str = ""

def parse_pdf(path: str) -> str:
    """
    Read all text from a PDF file.
    
    Args:
        path: Path to the PDF file.
    Returns:
        The concatenated text of all pages.
    """
    reader = PdfReader(path)
    return "".join(page.extract_text() or "" for page in reader.pages)

def summarize_sections(state: AgentState, llm: ChatGoogleGenerativeAI) -> AgentState:
    """
    Split the PDF text into chunks, summarize each section, and return bullet points.
    
    Args:
        state: AgentState containing `state.text`.
        llm: Initialized ChatGoogleGenerativeAI instance.
    Returns:
        Updated AgentState with `state.summary`.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    docs = splitter.create_documents([state["text"]])
    
    prompt_template = """
    You are a helpful assistant that summarizes research papers.
    Here are the document chunks:
    {text}
    
    Summarize each logical section in bullet points under its section name.
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = (prompt | llm | StrOutputParser())
    
    # Invoke the chain on the list of docs to get a structured summary.
    summary = chain.invoke({"text": docs})
    return AgentState(text=state["text"], summary=summary, script="")

def style_transfer(state: AgentState, llm: ChatGoogleGenerativeAI) -> AgentState:
    """
    Turn an academic summary into a conversational Host/Expert podcast script.
    
    Args:
        state: AgentState containing `state.summary`.
        llm: Initialized ChatGoogleGenerativeAI instance.
    Returns:
        Updated AgentState with `state.script`.
    """
    prompt = f"""
    You are a podcast host in dialogue with an expert. Podcast: "ChatCast".
    Rewrite this academic summary into a conversational Host/Expert script.
    Expert name: Michael Scott.

    {state["summary"]}
    """
    script = llm.predict(prompt)
    return AgentState(text=state["text"], summary=state["summary"], script=script)

def build_workflow(llm: ChatGoogleGenerativeAI) -> StateGraph:
    """
    Construct and compile the state graph for summarization → style transfer.
    
    Returns:
        A compiled StateGraph ready for invocation.
    """
    graph = StateGraph(AgentState)
    graph.add_node("summarize", lambda s: summarize_sections(s, llm))
    graph.add_node("style_transfer", lambda s: style_transfer(s, llm))
    graph.set_entry_point("summarize")
    graph.add_edge("summarize", "style_transfer")
    return graph.compile()

def generate_audio(script: str, murf_key: str, output_path: str):
    """
    Stream text-to-speech audio for each speaker line in the script.
    
    Args:
        script: The full Host/Expert script with lines prefixed by "**Host:**" or "**Michael Scott:**".
        murf_key: API key for Murf.
        output_path: File to append the audio chunks to.
    """
    client = Murf(api_key=murf_key)
    
    # Delete existing output file if it exists
    if os.path.exists(output_path):
        os.remove(output_path)
    
    # Split script into lines and process each speaker turn
    for line in script.split("\n"):
        if line.startswith("**Host:**"):
            voice = "en-US-natalie"
            text = line.split("**Host:**", 1)[1].strip()
        elif line.startswith("**Michael Scott:**"):
            voice = "en-US-terrell"
            text = line.split("**Michael Scott:**", 1)[1].strip()
        elif line.startswith("**Michael:**"):
            voice = "en-US-terrell"
            text = line.split("**Michael:**", 1)[1].strip()
        else:
            continue
        
        # Skip empty text
        if not text:
            continue
            
        # Stream and append audio
        stream = client.text_to_speech.stream(text=text, voice_id=voice)
        for chunk in stream:
            with open(output_path, "ab") as f:
                f.write(chunk)

def process_pdf_to_podcast(pdf_path: str, output_path: str = None):
    """
    Process a PDF into a podcast audio file.
    
    Args:
        pdf_path: Path to the input PDF.
        output_path: Path to save the output audio file. If None, uses the PDF name with .wav extension.
    
    Returns:
        Path to the generated audio file.
    """
    # Get API keys from environment
    google_api_key = os.getenv("GOOGLE_API_KEY")
    murf_api_key = os.getenv("MURF_API_KEY")
    
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    if not murf_api_key:
        raise ValueError("MURF_API_KEY environment variable not set")
    
    # Set default output path if not provided
    if output_path is None:
        output_path = os.path.splitext(pdf_path)[0] + ".wav"
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-8b", api_key=google_api_key)
    
    # Extract text from PDF
    raw_text = parse_pdf(pdf_path)
    if not raw_text.strip():
        raise ValueError("No text could be extracted from the PDF")
    
    # Process through workflow
    initial_state = AgentState(text=raw_text, summary="", script="")
    workflow = build_workflow(llm)
    final_state = workflow.invoke(initial_state)
    
    # Generate audio
    generate_audio(final_state["script"], murf_api_key, output_path)
    
    return output_path