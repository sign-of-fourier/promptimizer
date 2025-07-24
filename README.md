# Promptimizer

Promptimizer is a server that runs code optimization using the principles from the project DSPy. This implementation focuses on optimizing the instruction portion of the prompt.

It requires as input, two meta prompts to create the prompt (system and user), a text seperator for the to-be-written user prompt and the training data input, a sytem prompt to accompany the user prompt when training, labeled training data with an 'input' and 'output' columns and a label name to search for when extracting the answer from the output assuming it's in JSON format.





