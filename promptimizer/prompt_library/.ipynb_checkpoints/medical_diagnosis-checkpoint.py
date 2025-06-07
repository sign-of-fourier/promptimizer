writer_system = "You are a prompt engineer. Your job is to write a prompt to diagnose a patient based on his or her description of symptoms."

writer_user = """Write a prompt that instructs a language model to make a diagnosis based on the description of a disease. Tell the model that it will be given some text and instruct it to determine the disease based on the decription:

The model is to choose the disease from this list of choices: 
"cervical spondylosis", "impetigo", "urinary tract infection", "arthritis", "dengue", "common cold", "drug reaction", "fungal infection", "malaria", "allergy", "bronchial", "asthma","varicose veins", 'migraine', 'hypertension', 'gastroesophageal reflux disease', 'pneumonia", "psoriasis", "diabetes", "jaundice", "chicken pox", "typhoid", "peptic ulcer disease"

You are just writing the prompt. Do not leave a space for the disease description. I will label it and append it after your instructions.

Tell the model to give it's answer in JSON format with the labels: "rationale", "disease", and "confidence" in that order.
The probability should be chosen from one of these 5 possibilities only: "very confident", "confident", "neutral", "unconfident", "very unconfident".


Here is an example of properly formatted response:

{{"rationale": "The patient describes phlegm and sneezing. These are typical symptoms of the common cold. There are a couple of other things you could check to be sure, but it is most likely a cold.",
  "disease": "common cold",
  "confidence": "confident"
}}
"""
physicians_assistant = "You are a physician's assistant. Your job is to provide assistance in medical diagnosis based on a patient's symptoms."