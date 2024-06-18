from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from rich import print as rprint

from data_models import GraphState
from graph_utils import (retrieve,
                          grade_documents,
                          generate,
                          transform_query,
                          decide_to_generate,
                          grade_generation_vs_documents_and_question)


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


# Entrypoint as test
if __name__ == "__main__":
    load_dotenv()
    inputs = {"question": "Explain how the different types of agent memory work"}

    for output in app.stream(inputs):
        for key, value in output.items():
            rprint(f" NODE '{key}'".center(100, "="))
            rprint(f"[bold cyan]question:\n{value['question']}")
        print("\n" + "=" * 100 + "\n")

    # Final Generation
    rprint(f"[bold cyan]Answer:\n{value['generation']}")
