from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
from src.core import Rag

rag = Rag()
rag.create_dataset()
evaluation_dataset = EvaluationDataset.from_list(rag.test_datasets)
evaluator_llm = LangchainLLMWrapper(rag.llm)

result = evaluate(dataset=evaluation_dataset,metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],llm=evaluator_llm)
result