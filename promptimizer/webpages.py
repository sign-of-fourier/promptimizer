rag_help_page = """<html>
<title>Quante Carlo: RAG Instructions</title>
<h1>RAG File Preparation Instructions</h1>
<body>
<p>
<table border=0>
<tr><td>
     &nbsp;&nbsp;
    </td>
    <td colspan=2>
    <font size="+1"><br> &bull; A common way to implement rag, is to use the input to search for something such as a record in a vector db.
        You can use RAG as a kind of <a href="https://semi.supervised.com">semi-supervised learning.</font></a>
    </td><td>
     &nbsp;&nbsp;&nbsp;&nbsp;
    </td>
    </tr>

    <tr><td>
     &nbsp;&nbsp;
    </td>
    <td colspan=2><br><font size="+1">&bull; For example, in <a href="https://www.kaggle.com/competitions/llm-detect-ai-generated-text/data?select=train_essays.csv">this dataset</a>, there is a training set and a test set and each contains text and a sentiment label.
        The task is to use the data in the training set to attempt to label the data in the test set with the correct sentiment.
        </font<
    </td><td>
     &nbsp;&nbsp;&nbsp;&nbsp;
    </td></tr>
    <tr><td>
     &nbsp;&nbsp;
    </td>
    <td colspan=2><br><font size="+1">&bull; In order to run this as a <b>Machine Learning Prompt Optimization Task</b>, break the training set into two pieces.
        We'll call them <ol> <li><i>historical examples</i> </li>and the <li> <i>to-be-augmented training set</i></li></ol>  The idea is that given a piece of text from the <i>to-be-augmented training set</i>, look up records in the <i>historical examples set</i> and their corresponding labels.
        Put those records (text and label) into the prompt as <b><a href="https://fewshot.com">few shot</a></b> examples.
        Then the text in each record in your new <i>RAG training set</i> is a concatenation of the text from the <i>to-be-augmented training set</i> and some relevant selections from the <i>historical examples</i> with and their corresponding labels.
    </font>
    </td><td>
     &nbsp;&nbsp;&nbsp;&nbsp;
    </td></tr>
    <tr><td>
     &nbsp;&nbsp;
    </td>
    <td colspan=2><font size="+1"><br>
        &bull; Here is an example of an augmented record. The first portion is from the <i>to be augmented training set</i> and the rest is the augmentation from the <i>historical examples</i>.
    </font>
    </td><td>
     &nbsp;&nbsp;&nbsp;&nbsp;
    </td></tr>
    <tr>
        <td>
        &nbsp;&nbsp;
        </td>
        <td colspan=2>
        <hr> <br>

        </td>
        <td>
        </td>
    </tr>
    <tr><td>
     &nbsp;&nbsp
    </td>
    <td>
     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    </td>
    <td>
@ryanjreilly I love the irony of the 'heart shape' the officer's zip-tie cuffs make on his back. :-)<br>
<br><font color="green">
---------<br>
<br></font><font color="darkblue">
 ### EXAMPLE 1 ###</font><br><font color="darkred">
 @Harvard_Law @HarvardBLSA Let's End Police Brutality. Buy shirt at http://t.co/9tyHDKDF8C<br>
 TRUE LABEL: not a rumor</font><br>
<br><font color="darkblue">
 ### EXAMPLE 2 ###<br></font>
 <font color="darkred">@ryanjreilly at least the officers pictured here are wearing regular patrol uniforms, instead of looking like they're about to go to war.<br>
 TRUE LABEL: not a rumor</font><br>
<br>
<font color="darkblue">### EXAMPLE 3 ###</font><br>
<font color="darkred">@Tha_J_Appleseed Going for an officer's gun should...<br>
 TRUE LABEL: rumor</font>
    </td><td>
     &nbsp;&nbsp;&nbsp;&nbsp;
    </td></tr>
    <tr><td></td>
    <td colspan=2><font size="+1"><br>Notes<ol><li>The text from the original <i>to-be-augmented training set </i> is first and in black.</li>
    <li>Then a separator that I added of dashed lines in green: '<font color="green">---------</font>'</li>
    <li>Next, are the few shot examples. Each one is labeled as '<font color="darkblue">### EXAMPLE <i>N</i> ###</font>' where <i>N</i> is a ordinal. The labels are in dark blue and each historical example is in dark red.
    The color is not part of the prompt but shown here for clarity.</li></ul></font></td>
    <td></td></tr>
</table>
<br>
<a href="http://localhost:5000">Return</a>
<br>(c) Qaunte Carlo, 2025
<br>
</html>
"""



optimize_form = """<html>
<table border=1>
    <tr><td> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; </td>
        <td colspan=2>{}</td>
        <td>  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; </td></tr>
    <tr>
        <td></td>
        <form action="/optimize" method="POST" enctype="multipart/form-data">
              <td>
                  Training Data File
              </td>
              <td><input type="file" name="data">
                  {}
              </td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td>
                Name of output file (internal use)
              </td>
              <td>
                <input type="text" name="filename_id" value="{}">
              </td>
        <td></td>
    </tr>
    <tr>
        <td></td>
                <td>
                  Key Path (internal use)
                </td>
                <td>
                  <input type="text" name="key_path" value={}>
                </td>
        <td></td>
    </tr>
    <tr>
                <td></td>
                <td>
                    <input type="submit" value="Optimize!"></input>
                </td>
                <td></td>
        <td></td>
    </tr>

            </form>
</table>
"""



check_status_form = """<html><title>Quante Carlo</title><br><body><p>
<form action="/check_status?use_case={}&next_action={}" method="POST" enctype="multipart/form-data">
<br>
<table border=1>
    <tr>
        <td> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; </td>
        <td colspan=2>{}</td>
        <td> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; </td>
    </tr>
  <tr>
    <td></td>
    <td>
        jobArn
    </td>
    <td>
        {}
        <input type="text" name="jobArn" value="{}"></input>
    </td>
    <td></td>
  </tr>
  <tr>
      <td></td>
    <td>
        Key Path
    </td>
    <td>
      <input type="text" name="key_path" value="{}"></input>
    </td>
    <td></td>
  </tr>
  <tr>
   <td></td>
    <td>
        Filename
    </td>
    <td>
        <input type="text" name="filename_id" value="{}"></input>
    </td>
    <td></td>
  </tr>
    <tr>
        <td></td>
        <td></td>
        <td>
            <input type="submit" value="Check Status"></input>
        </td>
    <td></td>
    </tr>
    </table>
</form>

"""



enumerate_prompts =  """
<br>
<form action="/enumerate_prompts?use_case={}&deployment={}" method="POST"  enctype="multipart/form-data">
<table border=0>
    <tr>
        <td></td>
        <td colspan=2>Here you design your Meta Prompt, the prompts that will write candidates for your ideal prompt.</td>
        <td></td>
    </tr>
    <tr>
        <td> &nbsp; &nbsp; &nbsp; </td>
        <td><b>Meta Prompt - System</b></td>
        <td><textarea name="writer_system" rows=3 cols=60>{}</textarea></td>
        <td> &nbsp; &nbsp; &nbsp; </td>
    </tr>
    <tr>
        <td></td>
        <td><b>Meta Prompt - User</b></td>
        <td><textarea name="writer_user" rows=8 cols=100>{}</textarea></td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td><b>Seperator</b><br> This will be used when adding your data to the prompt.</td>
        <td><input type="text" name="separator" rows=3 value="{}"></input></td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td stype="width:60px"><b>Task System</b> Accompanies the prompt to be written.</td>
        <td><input type="text" name="task_system" rows=3 value="{}"></input></td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td><b>JSON key</b> for label. Should match the prompt.</td>
        <td><input width=70 type="text" name="label" value="{}"></input</td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td><b>Demonstrations </b>when enumerating the space. <br> Currently, only implemented for defect_detection.</td>
        <td><input type="file" name="demonstrations"></td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td><b>Evaluation method</b> for label. Should match the prompt.</td>
        <td><select name="evaluator">
            <option value="accuracy">Accuracy</option>
            <option valuie="AUC">AUC</option>
            </select>
        </td>
        <td></td>
    </tr>

    <tr>
        <td></td>
        <td><b>Model</b></td>
        <td>
            <table border=1>
                <tr>
                    <td>
                        <u>Model Name</u>
                    </td>
                    <td>
                        <u>N Prompts</u>
                    </td>
                    <td> &nbsp; &nbsp; </td>
                    <td> <u> Model Name </u>
                    </td>
                    <td>
                       <u> N Prompts </u>
                    </td>
                </tr>
                {}

            </table>
        </td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td>Password</td>
        <td><input name="password" type="text"></input>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td><input type=submit value=submit></input>
        <td></td>
    </tr>
</table>
</form>
"""

use_case_selector = """
<html>
<body><br>
<table border=0>
    <tr>
        <td>
            &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
        </td>
        <td>
            <h1>Welcome to the Promptimizer</h1>
        </td>
        <td align='right'>by Quante Carlo

        </td>
        <td>
            &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
        </td>
    </tr>
    <tr><td colspan=4> &nbsp; </td></tr>
    <tr>
        <td>
            &nbsp; &nbsp; &nbsp; &nbsp;
        </td>
        <td colspan=2>
            <h2>Select Use Case</h2>
            <table border=0>
                <tr>
                    <td>
                        <a href="/prompt_preview?use_case=medical_diagnosis&deployment=bedrock">Medical Diagnosis</a>
                    </td>
                    <td>
                        Diagnose a patient based on text describing his or her symptoms.
                    </td>
                    <td>
                        <a href="https://https://huggingface.co/datasets/gretelai/symptom_to_diagnosis">Hugging Face</a>
                    </td>
                </tr>
                <tr>
                    <td>
                        <a href="/prompt_preview?use_case=ai_detector&deployment=bedrock">AI Detector</a>
                    </td>
                    <td>
                        Given some text, determine if the text was generated by a human or a language model.
                    </td>
                    <td>

                    <a href="https://www.kaggle.com/competitions/llm-detect-ai-generated-text/data?select=train_essays.csv">Kaggle</a>
                    </td>
                </tr>
                <tr>
                    <td>
                        <a href="/prompt_preview?use_case=defect_detector&deployment=azure">Defect Detection</a>
                    </td>
                    <td>
                       Given an image determine if the part is defective. <br>Can also be used to detect AI generated images. Important for fraud detection.
                    </td>
                    <td>
                       <a href="https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product">Kaaggle</a>
                    </td>
                </tr>
            </table>
       </td>
       <td>
           &nbsp; &nbsp; &nbsp; &nbsp;
        </td>
        </tr>
        <tr>
        <td></td>
        <td colspan=2><hr></td>
        <td></td>
        </tr>
        <tr>
        <td></td>
        <td colspan=2>
            <ul>
                <li> <font size="+1">The input file needs to have two columns labeled 'input' and 'output'.</li>
                <li>If you're using RAG, prepare the input file <a href="/rag">accordingly.</a></li>
                <li>There are three kinds of evaluators:
                <ol><li>Accuracy - If target matches or not</li>
                    <li>AUC - probability must be from the following list: <i>'very unlikely', 'unlikely', 'equally likely and unlikely', 'likely', 'very likely'</i></li>
                    <li>AI prompt - there will be an additional prompt that evaluates the input and the answer and gives a 'correct' or 'incorrect' verdict.</li></ol></li>
            </ul>
            </font>
        </td>
        <td>
            &nbsp; &nbsp; &nbsp; &nbsp;
        </td>
    </tr>
</table>


"""

