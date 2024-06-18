from rich import print as rprint

from chains import (rag_chain,
                     retrieval_grade_chain,
                     question_rewriter_chain,
                     hallucination_grade_chain,
                     answer_grade_chain)
from data_models import GraphState
from retrieval import retriever


# NODES FUNCTIONS
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
    print(" CHECK RELEVANCE TO QUESTION ".center(100, "-"))
    question = state["question"]
    documents = state["documents"]

    # Score each document
    filtered_docs = []
    for d in documents:
        score = retrieval_grade_chain.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            rprint(
                "[green] GRADE: DOCUMENT IS RELEVANT [/green]".center(100, "-"))
            filtered_docs.append(d)
        else:
            print(
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
    

# CONDITIONAL EDGES FUNCTIONS
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

    score = hallucination_grade_chain.invoke(
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
        score = answer_grade_chain.invoke({"question": question, "generation": generation})
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
