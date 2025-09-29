meta_system = "You are a prompt engineer. Your job is to write a prompt to diagnose a patient based on his or her description of symptoms."

meta_user = """Write a prompt that instructs a language model to make a diagnosis based on a patient's description of symptoms. Tell the model that it will be given some text and instruct it to determine the disease based on the description:

The model is to choose the disease from this list of choices: 
"cervical spondylosis", "impetigo", "urinary tract infection", "arthritis", "dengue", "common cold", "drug reaction", "fungal infection", "malaria", "allergy", "bronchial", "asthma", "varicose veins", 'migraine', 'hypertension', 'gastroesophageal reflux disease', 'pneumonia", "psoriasis", "diabetes", "jaundice", "chicken pox", "typhoid", "peptic ulcer disease"

You are just writing the prompt. Do not leave a space for the disease description. I will label the patient's description and append it after your instructions.

Tell the model to give it's answer in JSON format with the labels: "rationale", "disease", and "confidence" in that order.
The probability should be chosen from one of these 5 possibilities only: "very confident", "confident", "neutral", "unconfident", "very unconfident".

Do not introduce your prompt or provide any other text.

"""
separator = "### PATIENTS SYMPTOMS ###"
task_system = "You are a physician's assistant. Your job is to provide assistance in medical diagnosis based on a patient's symptoms."
label_name="disease"
evaluator_prompt = ''
