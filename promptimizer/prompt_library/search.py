meta_system = "You are a filing clerk. Your job is to determine the relevance of an image with respect to a query."


meta_user = """I'm going to give you a query and an image. Rate the relevance of the text to figuring out the answer to the question. 
Give your relevance from the following list: "critical relevance", "very relevant", "somewhat relevant", "not very relevant", "completely irrelevant"
Give your rationale and ranking in that order in JSON format like this:
{"rationale": "The text says nothing that would be useful in answering the question'",
 "relevance": "completely irrelevant"}
"""

task_system = "Children playing games outside, preferably somewhere other than the United States."
separator = '### IMAGE ###'
#task_system = "I'm going to ask you a question. Before I give you the question, I am going to also give you some reference material. The reference material is based on a search of a corpus. It may or may not be relevant. It may help you answer the question."
label_name="relevance"
evaluator_prompt = "I'm going to give you a goal of an ad campaign and an image. Give me caption that goes with image that fits the goal of the ad campaign. Do not introduce your text or give any other context. Only give me your caption text."

