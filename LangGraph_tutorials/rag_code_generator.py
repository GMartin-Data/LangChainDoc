from typing import TypedDict

from bs4 import BeautifulSoup as Soup
from dotenv import load_dotenv
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, StateGraph
from rich import print as rprint


# SOURCE ENVIRONMENT
load_dotenv()

# RAG
## LOAD DOCUMENTS
URL = "https://python.langchain.com/v0.1/docs/expression_language/"
loader = RecursiveUrlLoader(
    url = URL,
    max_depth = 20,
    extractor = lambda page: Soup(page, "html.parser").text
)
docs = loader.load()
d_sorted = sorted(docs, key=lambda doc: doc.metadata["source"])
d_reversed = list(reversed(d_sorted))
concatenated_content = "\n\n\n --- \n\n\n".join(
    [doc.page_content for doc in d_reversed]
)
documents = [Document(page_content=concatenated_content)]

## SPLIT DOCUMENTS
embeddings = OpenAIEmbeddings()
text_splitter = SemanticChunker(embeddings=embeddings)
split_docs = text_splitter.split_documents(documents)

## CREATE VECTORSTORE AND RETRIEVER
vectorstore = Chroma.from_documents(
    documents = split_docs,
    embedding = embeddings
)
retriever = vectorstore.as_retriever()


# CHAIN
## PROMPT TEMPLATE
code_gen_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a coding assistant with expertise in LCEL, LangChain expression language. \n
        Here is a full set of LCEL documentation: \n ------- \n {context} \n ------- \n Answer the user
        question based on the above provided documentation. Ensure any code you provide can be executed \n
        with all required imports and variables defined. Structure your answer with a description of the code solution. \n
        Then list the imports. And finally list the functioning code block. Here is the user question:""",
     ),
     ("placeholder", "{messages}"),
])

## LLM
llm = ChatOpenAI(temperature=0, model="gpt-4o")

## DATA MODEL (LLMs WISHED OUTPUT)
class Code(BaseModel):
    """Code Output"""
    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")
    description: str = Field(description="Schema for code solutions to questions about LCEL")
    
## CHAIN COMPONENTS
code_gen_chain = code_gen_prompt | llm.with_structured_output(Code)


# GRAPH
## STATE
class GraphState(TypedDict):
    """
    Represents the state of our graph

    Attributes:
        error: Binary flag for control flow to indicate whether test error was tripped
        messages: With user question, error messages, reasoning
        generation: Code solution
        iterations: Number of tries
    """
    error: str
    messages: list
    generation: Code
    iterations: int

## PARAMETERS
max_iterations = 3
flag = "do not reflect"

## NODES FUNCTIONS
def generate(state: GraphState) -> GraphState:
    """
    Generate a code solution
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): New key added to state, generation
    """
    # Retrieve state
    messages = state["messages"]
    iterations = state["iterations"]
    error = state["error"]
    # We have been routed back to generation with an error
    if error == "yes":
        messages += [(
            "user",
            "Now, try again. Invoke the code tool to structure the output with a prefix, imports, and code block:"
        )]   
    # Solution
    code_solution = code_gen_chain.invoke({
        "context": retriever,
        "messages": messages
    })
    messages += [(
        "assistant",
        f"{code_solution.prefix} \n Imports: {code_solution.imports} \n Code: {code_solution.code}"
    )]
    # Increment
    iterations += 1
    return {
        "generation": code_solution,
        "messages": messages
    }
    
def code_check(state: GraphState) -> GraphState:
    """
    Check code
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): New key added to the state: error
    """
    rprint("[bold yellow] 🔎 CHECKING CODE 🔎 [/bold yellow]".center(126, '-'))
    # Retrieve state
    messages = state["messages"]
    code_solution = state["generation"]
    iterations = state["iterations"]
    # Get solution components
    imports = code_solution.imports
    code = code_solution.code
    # Check imports 🔴
    try:
        exec(imports)
    except Exception as e:
        rprint("[bold red]🔴 CODE IMPORT CHECK: FAILED 🔴[/bold red]".center(126, "-"))
        error_message = [("user", f"Your solution failed the import test: {repr(e)}")]
        messages += error_message
        return {
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations,
            "error": "yes"
        }
    # Check execution 🟠
    try:
        exec(imports + "\n" + code)
    except Exception as e:
        rprint("[bold orange] CODE BLOCK CHECK: FAILED [/bold orange]".center(126, "-"))
        error_message += [("user", f"Your solution failed the code test: {e}")]
        return {
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations,
            "error": "yes",
        }
    # No errors 🟢
    rprint(" NO CODE TEST FAILURES ".center(50, "🟢"))
    return {
        "generation": code_solution,
        "messages": messages,
        "iterations": iterations,
        "error": "no",
    }

def reflect(state: GraphState) -> GraphState:
    """
    Reflect on errors
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): New key added to state, generation
    """
    # Retrieve state
    messages = state["messages"]
    iterations = state["iterations"]
    code_solution = state["generation"]
    # Prompt reflexion
    messages += [("user", "Review your previous answer and find problems with your answer")]
    # Add reflexion
    reflections = code_gen_chain.invoke({
        "context": retriever, "messages": messages
    })
    messages += [("assistant", f"Here are reflections on the error: {reflections}")]
    rprint("[bold magenta] 🧠 REFLEXION PERFORMED 🧠[/bold magenta]".center(126, "."))
    return {
        "generation": code_solution,
        "messages": messages,
        "iterations": iterations
    }
    
## EDGES FUNCTIONS
def decide_to_finish(state: GraphState) -> str:
    """
    Determines whether to finish.
    Args:
        state (dict): The current graph state
    Returns:
        str: Next node to call
    """
    # Retrieve the state
    error = state["error"]
    iterations = state["iterations"]
    # Decide
    if (error == "no") or (iterations == max_iterations):
        rprint("[bold white]🏁 DECISION: FINISH 🏁[/bold white]".center(126, "="))
        return "end"
    else:
        rprint("[bold white]♻️ DECISION: RE-TRY SOLUTION ♻️[/bold white]".center(126, "="))
        if flag == "reflect":
            return "reflect"
        else:
            return "generate"

## ASSEMBLE
### INSTANCIATE AND LINK TO STATE
workflow = StateGraph(GraphState)
### DEFINE NODES
workflow.add_node("generate", generate)  # generation solution
workflow.add_node("check_code", code_check)  # check code
workflow.add_node("reflect", reflect)  # reflect
### DEFINE EDGES
workflow.add_edge("generate", "check_code")
workflow.add_conditional_edges(
    "check_code",
    decide_to_finish,
    {
        "end": END,
        "reflect": "reflect",
        "generate": "generate",
    },
)
workflow.add_edge("reflect", "generate")
### SET ENTRY POINT
workflow.set_entry_point("generate")
### COMPILE
app = workflow.compile()