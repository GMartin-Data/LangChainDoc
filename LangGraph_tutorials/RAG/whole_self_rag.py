from typing import List

from dotenv import load_dotenv
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, StateGraph
from rich import print as rprint
from typing_extensions import TypedDict


# SOURCE ENVIRONMENT
load_dotenv()

# INSTANCIATE BASE LLM
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

# RETRIEVER
## LOAD DOCUMENTS
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item
             for doc in docs
             for item in doc]  # Flatten docs list
## SPLIT DOCUMENTS
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
docs_splits = text_splitter.split_documents(docs_list)
## ADD TO VECTOR DB AND ASSOCIATE RETRIEVER
vectorstore = Chroma.from_documents(
    documents = docs_splits,
    collection_name = "rag-chroma",
    embedding = OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()

# RETRIEVAL GRADER CHAIN
## BUILD PROMPT
retrieval_grader_system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

retrieval_grader_human = "Retrieved document: \n\n {document} \n\n User question: {question}"

retrieval_grader__prompt = ChatPromptTemplate.from_messages([
    ("system", retrieval_grader_system),
    ("human", retrieval_grader_human)
])
## STRUCTURE LLMs OUTPUT
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents"""

    binary_score: str = Field(
        description = "Documents are relevant to the question, 'yes' or 'no'"
    )
retrieval_grader_llm = llm.with_structured_output(GradeDocuments)
## CHAIN
retrieval_grader_chain = retrieval_grader__prompt | retrieval_grader_llm

# GENERATE CHAIN
rag_prompt = hub.pull("rlm/rag-prompt")
rag_chain = rag_prompt | llm | StrOutputParser()

# HALLUCINATION GRADER CHAIN
## BUILD PROMPT
hallucination_grader_system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""

hallucination_grader_human = "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"

hallucination_grader_prompt = ChatPromptTemplate.from_messages([
    ("system", hallucination_grader_system),
    ("human", hallucination_grader_human)
])
## STRUCTURE LLMs OUTPUT
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description = "Answer is grounded in the facts, 'yes' or 'no'"
    )
hallucination_grader_llm = llm.with_structured_output(GradeHallucinations)
## CHAIN
hallucination_grader_chain = hallucination_grader_prompt | hallucination_grader_llm

# ANSWER GRADER CHAIN
## BUILD PROMPT
answer_grader_system = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""

answer_grader_human = "User question: \n\n {question} \n\n LLM generation: {generation}"

answer_grader_prompt = ChatPromptTemplate.from_messages([
    ("system", answer_grader_system),
    ("human", answer_grader_human)
])
## STRUCTURE LLMs OUTPUT
class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description = "Answer addresses the question, 'yes' or 'no'"
    )
answer_grader_llm = llm.with_structured_output(GradeAnswer)
## CHAIN
answer_grader_chain = answer_grader_prompt | answer_grader_llm

# QUESTION REWRITER CHAIN
## BUILD PROMPT
question_rewriter_system = """You a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""

question_rewriter_human = "Here is the initial question: \n\n {question} \n Formulate an improved question."

question_rewriter_prompt = ChatPromptTemplate.from_messages([
    ("system", question_rewriter_system),
    ("human", question_rewriter_human)
])
## CHAIN
question_rewriter_chain = question_rewriter_prompt | llm | StrOutputParser()

# DEFINE GRAPH
## STATE
class GraphState(TypedDict):
    """
    Represents the state of the graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]

## NODES
def retrieve(state: GraphState) -> GraphState:
    """
    Retrieve documents.

    Args:
        state (dict): The current graph state
    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    rprint(
        "[bold white] RETRIEVE [/bold white]".center(100, "-"))
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    return {
        "question": question,
        "documents": documents,
    }
    
def generate(state: GraphState) -> GraphState:
    """
    Generate answer.

    Args:
        state (dict): The current graph state
    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    rprint(
        "[bold white] GENERATE [/bold white]".center(100, "-"))
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {
        "question": question,
        "documents": documents,
        "generation": generation
    }
    
def grade_documents(state: GraphState) -> GraphState:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state
    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """
    rprint(
        "[bold white] CHECK RELEVANCE TO QUESTION [/bold white]".center(100, "-"))
    question = state["question"]
    documents = state["documents"]

    # Score each document
    filtered_docs = []
    for d in documents:
        score = retrieval_grader_chain.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            rprint(
                "[green] GRADE: DOCUMENT IS RELEVANT [/green]".center(100, "-"))
            filtered_docs.append(d)
        else:
            rprint(
                "[red] GRADE: DOCUMENT IS NOT RELEVANT [/red]".center(100, "-"))
            continue  # Optional but meaningful

    return {
        "question": question,
        "documents": filtered_docs
    }
    
def transform_query(state: GraphState) -> GraphState:
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state
    Returns:
        state (dict): Updates question key with a re-phrased question
    """
    rprint(
        "[bold orange] TRANSFORM QUERY [/bold orange]".center(100, "-"))
    question = state["question"]
    documents = state["documents"]

    # Re-write question
    better_question = question_rewriter_chain.invoke({"question": question})
    return {
        "question": better_question,
        "documents": documents
    }
    
## CONDITIONAL EDGES
def decide_to_generate(state: GraphState) -> str:
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state
    Returns:
        str: Binary decision for next node to call 
    """
    rprint(
        "[bold yellow] ASSESS GRADED DOCUMENTS [/bold yellow]".center(100, "-"))
    filtered_documents = state["documents"]

    if not filtered_documents:
        # No document has been found relevant
        # We will re-generate a new query
        rprint(
            "[red] DECISION: NO DOCUMENT IS RELEVANT TO QUERY [/red]".center(100, "-"))
        return "transform_query"
    else:
        # There are relevant documents
        rprint(
            "[green] DECISION: GENERATE [/green]".center(100, "-"))
        return "generate"
    
def grade_generation_vs_documents_and_question(state: GraphState) -> str:
    """
    Determines whether the generation:
    - is grounded in the documents,
    - answers question.

    Args:
        state (dict): The current graph state
    Returns:
        str: Decision for the next node to call
    """
    print(
        "[bold yellow] CHECK HALLUCINATIONS [/bold yellow]".center(100, "-"))
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader_chain.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score.binary_score

    # Check hallucinations
    if grade == "yes":
        rprint(
            "[green] DECISION: GENERATION IS GROUNDED IN DOCUMENTS [/green]".center(100, "-"))
        # Check question answering
        rprint(
            "[bold yellow] GRADE GENERATION VS QUESTION [/bold yellow]".center(100, "-"))
        score = answer_grader_chain.invoke({"question": question, "generation": generation})
        grade = score.binary_score
        if grade == "yes":
            rprint(
                "[green] DECISION: GENERATION ADDRESSES QUESTION [/green]".center(100, "-"))
            return "useful"
        else:
            rprint(
                "[orange] DECISION: GENERATION DOES NOT ADDRESS QUESTION [/orange]".center(100, "-"))
            return "not_useful"
    else:
        rprint("[red] DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS [/red]".center(100, "-"))
        return "not supported"
    
# ASSEMBLE GRAPH
workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)

# Build Graph
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",  # Starting node
    decide_to_generate,  # Decision function
    # Map outputs of decision function to destination node's name
    {
        "transform_query": "transform_query",
        "generate": "generate"
    }
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges(
    "generate",
    grade_generation_vs_documents_and_question,
    {
        "not supported": "generate",
        "not useful": "transform_query",
        "useful": END
    }
)

# Compile
app = workflow.compile()

# RUN
inputs = {"question": "Anthropic GPT agent tools"}

for output in app.stream(inputs):
    for key, value in output.items():
        rprint(f" NODE '{key}'".center(100, "="))
        rprint(f"[bold cyan]question:\n{value['question']}")
    print("\n" + "=" * 100 + "\n")

# Final Generation
rprint(f"[bold cyan]Answer:\n{value['generation']}")


# # ENTRYPOINT: TESTS
# if __name__ == "__main__":
#     question = "agent memory"
#     docs = retriever.invoke(question)
    
#     # ## TEST RETRIEVAL GRADER
#     # for idx, doc in enumerate(docs):
#     #     rprint(f" Chunk {idx} ".center(100, "#"))
#     #     rprint(doc.page_content)
#     # for idx, doc in enumerate(docs):
#     #     doc_txt = doc.page_content
#     #     print(f" Relevance test for chunk {idx} ".center(150, "#"))
#     #     grade = retrieval_grader_chain.invoke({"question": question, "document": doc_txt})
#     #     print("Is the current chunk relevant for the question? ", grade)
        
#     # ## TEST GENERATE CHAIN
#     # rprint(rag_prompt)
#     # generation = rag_chain.invoke({"context": docs, "question": question})
#     # rprint(generation)
    
  
    
    
    