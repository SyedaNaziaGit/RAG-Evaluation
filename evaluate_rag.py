import os
from google import genai
from typing import Any
from ragas.embeddings.base import embedding_factory
from ragas import EvaluationDataset,evaluate

# (question, ground_truth_reference)
QA_PAIRS = [
    (
        "What does sustainable development mean?",
        "Development that meets today's needs without preventing future generations from meeting their own needs",
    ),
    (
        "What are the three pillars of sustainable development?",
        "Economic growth, social inclusion, and environmental protection",
    ),
    (
        "How many global goals did countries agree on in the 2030 Agenda?",
        "17 Sustainable Development Goals",
    ),
    (
        "What is the Paris Agreement's temperature target?",
        "Limit global warming to 1.5 degrees Celsius above pre-industrial levels",
    ),
    (
        "What does a circular economy try to reduce?",
        "Waste, by keeping products and materials in use for as long as possible",
    ),
    (
        "What are two examples of circular economy actions?",
        "Recycling and repairing products instead of throwing them away",
    ),
    (
        "What basic needs are highlighted under social equity?",
        "Access to clean water, basic education, and healthcare for all people",
    ),
    (
        "What role do individuals play in sustainable development?",
        "Individuals can reduce waste, save energy, and make eco-friendly choices",
    ),
    (
        "Why is limiting global warming to 1.5 degrees Celsius important?",
        "Beyond 1.5 degrees Celsius, risks of floods, droughts, and extreme weather increase significantly",
    ),
    (
        "What is the target year for the Sustainable Development Goals?",
        "2030",
    ),
]

def run_rag_evaluation(chain , retriever):
    google_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    # Configuring the embedding factory for Google's provider
    embeddings = embedding_factory(
        provider="google",
        model="text-embedding-004",
        client=google_client
    )
    #creating dataset of dictonary in list
    dataset : list[dict[str,Any]] = []
    
    for each,(question,ground_truth) in enumerate(QA_PAIRS):
        context_docs = retriever.invoke(question)
        contexts = [doc.page_content  for doc in context_docs]
        context_str = "\n\n".join(contexts)
        result = chain.invoke({"context":context_str,"question":question})
        response = result.content
        
        dataset.append({
            "user_input":question,
            "retrieved_contexts": contexts,
            "response":response,
            "reference":ground_truth,
        })    
        evaluation_dataset = EvaluationDataset.from_list(dataset)
        results = evaluate(
            dataset = evaluation_dataset,
            embeddings = embeddings
        )
        df = results.to_pandas()
        #save dataframe df to csv file
        df.to_csv("evaluation_results.csv",index = False)