# Promptimizer

Promptimizer is a server that runs code optimization using the principles from the project DSPy. This implementation focuses on optimizing the instruction portion of the prompt.

It requires as input, two meta prompts to create the prompt (system and user), a text seperator for the to-be-written user prompt and the training data input, a sytem prompt to accompany the user prompt when training, labeled training data with an 'input' and 'output' columns and a label name to search for when extracting the answer from the output assuming it's in JSON format.

There are examples to help explain.
1. Select an example
2. Modify the prompts as needed
3. Select models that create the search space. You must select at least 100 each. 
3. Click Submit'. After the search space is created.

- There are links pointing to where the training datasets come from. The columns may need to be relabeled as 'input' and 'output'. Also, they should be downsampled because the original files are too large.





