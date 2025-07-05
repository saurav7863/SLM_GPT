from langchain.llms import LlamaCpp
from langchain.agents import Tool, initialize_agent, AgentType
from tools.pdf_tool import fill_pdf_form
from tools.web_search import web_search, open_url

# Load the local LLM model
model_path = "models/mistral-7b-q4.gguf"  # path to the quantized model file (GGUF/GGML)
# If using a GGUF model, LlamaCpp will load it. Adjust n_ctx if needed.
llm = LlamaCpp(
    model_path=model_path,
    n_ctx=4096,
    n_gpu_layers=-1,   # offload all layers to GPU (Metal) if possible
    n_threads=4,       # use 4 threads for CPU (adjust per your CPU cores)
    f16_kv=True,       # use half-precision for KV cache to save memory
    temperature=0.7,   # a moderate temperature for balanced creativity
    max_tokens=1024,   # limit max tokens in the response
    verbose=False
)

# Define the tools for the agent
tools = [
    Tool(
        name="Search",
        func=web_search,
        description="Use this to search the web for up-to-date information on a query."
    ),
    Tool(
        name="OpenURL",
        func=open_url,
        description="Use this to retrieve and read the content of a webpage by URL (after searching for it)."
    ),
    Tool(
        name="FillPDF",
        func=fill_pdf_form,
        description="Use this to fill out a PDF form with provided values. Input format: 'Field1=Value1; Field2=Value2; ...'."
    )
]

# Initialize the agent with the tools and LLM
agent_chain = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=False
)