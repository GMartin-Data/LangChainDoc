from langchain_core.prompts import ChatPromptTemplate

# UTILITY FUNCTION
def get_prompt(system, human):
    return ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", human)
    ])

# RETRIEVAL GRADE
retrieval_grade_system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

retrieval_grade_human = "Retrieved document: \n\n {document} \n\n User question: {question}"
retrieval_grade_prompt = get_prompt(retrieval_grade_system, retrieval_grade_human)

# HALLUCINATION GRADE
hallucination_grade_system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""

hallucination_grade_human = "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"
hallucination_grade_prompt = get_prompt(hallucination_grade_system, hallucination_grade_human)

# ANSWER GRADE
answer_grade_system = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""

answer_grade_human = "User question: \n\n {question} \n\n LLM generation: {generation}"
answer_grade_prompt = get_prompt(answer_grade_system, answer_grade_human)

# QUESTION REWRITER
question_rewriter_system = """You a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""

question_rewriter_human = "Here is the initial question: \n\n {question} \n Formulate an improved question."
question_rewriter_prompt = get_prompt(question_rewriter_system, question_rewriter_human)
