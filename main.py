from dotenv import load_dotenv

load_dotenv()

from rag_pipeline import  buildRAGChain
from evaluate_rag import run_rag_evaluation

def main():
    print("BUILING RAG PIPELINE..")
    chain , retriever = buildRAGChain()
    print("Pipeline is Ready..\n RUNNIG EVALUATION..\n")
    run_rag_evaluation(chain , retriever)
    

if __name__ == "__main__":
    main()