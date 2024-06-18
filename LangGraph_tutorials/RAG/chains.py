from dotenv import load_dotenv
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

from .data_models import GradeDocuments, GradeHallucinations, GradeAnswer
from .prompts import (retrieval_grade_prompt,
                      hallucination_grade_prompt,
                      answer_grade_prompt,
                      question_rewriter_prompt)
from .retrieval import retriever

# Environment should be sourced somewhere else
load_dotenv()

# INSTANCIATE LLM
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

# RAG CHAIN
## Pull rag prompt
prompt = hub.pull("rlm/rag-prompt")
## Build convenient function
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
## Build chain
rag_chain = (prompt 
             | llm 
             | StrOutputParser 
             | RunnableLambda(format_docs)
             )

##### GRADERS CHAINS #####

# RETRIEVAL GRADER CHAIN
## Structure LLM's output
retrieval_grade_llm = llm.with_structured_output(GradeDocuments)
## Build Chain
retrieval_grade_chain = retrieval_grade_prompt | retrieval_grade_llm

# HALLUCINATION GRADER CHAIN
## Structure LLM's output
hallucination_grade_llm = llm.with_structured_output(GradeHallucinations)
## Build chain
hallucination_grade_chain = hallucination_grade_prompt | hallucination_grade_llm

# ANSWER GRADER CHAIN
## Structure LLM's output
answer_grade_llm = llm.with_structured_output(GradeAnswer)
## Build chain
answer_grade_chain = answer_grade_prompt | answer_grade_llm

# QUESTION REWRITER CHAIN
question_rewriter_chain = (question_rewriter_prompt
                           | llm
                           | StrOutputParser()
                           )
  