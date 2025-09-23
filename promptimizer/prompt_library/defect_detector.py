meta_system = "You are a prompt engineer. Your job is to prompts for a language model that produce accurate responses."
meta_user = """Look in these images of castings. Each images shows a casting of a submersible pump impeller. The first two are not defective and the second two are defective. Your job is to write a prompt that instructs a language model in how to identify a defective casting. 
Another language model will be given your prompt and an image of a casting. Write a prompt that tells the other language model how to determine if a casting is defective.
Tell the language model to rate the probability that the casting is defective using terms from the following 5 possibilities: 'very unlikely', 'likely', 'equally likely' and unlikely', likely, 'very likely'. Tell the language model to put the probability and rationale in JSON format like this:
{
"rationale": "There appears to be damage around the edges of the casting. Although it is warped, that is not a condition of a defect.",
"probability": "likely"
}
evaluator_prompt = ''
Do not introduce your prompt.
Do not label where the image should go. I will be appended to your instructions with its own label.
Do not provide any other text or formatting. Only provide the prompt.
"""
label_name = "probability"
separator = "Not used for images"
task_system = 'You are a quality control inspector. Your job is to determine if a casting is defective by looking at an image.'
