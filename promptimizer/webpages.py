
hidden = "<input type=\"hidden\" name=\"{}\" value=\"{}\"></input>\n"


tworows = "<tr><td><b>{}</b></td><td>{}</td></tr>\n"

threerows = "<tr><td><b>{}</b></td><td>{}</td><td>{}</td></tr>\n"


choose_product = """
{}

<body>
{}
<div class="column row"></div>

<div class="column tenth"></div>

<div class="column fifth">
  <div class="card">
  <font size=+3><b>50 Credits</b></font><br>
  <br><br>
  <font size="+3">$39</font><br>
  <form action='https://buy.stripe.com/test_8x23cvcVz1V831dfEwbjW00'>
  <input type="submit" name="product" value="Checkout">
  </form>
  </div>
</div>

<div class="column tenth"></div>

<div class="column fifth">
  <div class="card"><font size="+3">
    <b>Subscription</b></font><br>
    100 credits per month<br><br>
    <font size="+3">$59</font><br>
    <form action='stripe'>
    <input type="submit" name="product" value="Checkout">
    </form>
  </div>
</div>

<div class="column tenth"></div>

<div class="column fifth">
  <div class="card">
  <font size="+3"><b>Enterprise</b><br></font>
  Custom<br><br>
  <font size="+2">Starting at $1000</font><br>
  <form action='/email'>
  <input type="submit" name="product" value="Contact Us">
  </form>
  </div>
</div>


<div class="column tenth"></div>


</html>
"""




rag_help_page = """<html>

{}
<body>
<h1>RAG File Preparation Instructions</h1>
{}
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



waiting = """<html>
{}
<body>
{}
<div class="column row"></div>
<div class="column left">{}</div>
  <div class="column middle">
    <div class="card">
{}
    <form action="/check_status?use_case={}&next_action={}" method="POST">
{}
    <input type="submit" Value="Check Again"></input>
    </form>
    </form>

  </div>
</div>
<div class="column small"></div>





</body>
</html>
"""


optimize_form = """<html>
{}
<body>
{}
<div class="column left">
{}
</div>
<div class="column middle">
<table border=0>
    <tr><td> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; </td>
        <td colspan=2>{}</td>
        <td>  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; </td></tr>
    <tr>
        <form action="/optimize?use_case={}" method="POST" enctype="multipart/form-data">
        {}
    </tr>
        <tr>
        <td></td>
              <td>
                  Training Data File
              </td>
              <td><input type="file" name="data"></input>
              </td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td><b>Separator</b>Separates prompt and obserrvation.</td>
        <td><input width=70 type="text" name="separator" value="{}"></input</td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td><b>Task System</b> Accompanies the prompt to be written.</td>
        <td><input type="text" name="task_system" rows=3 value="{}"></input></td>
        <td></td>
    </tr>

    <tr>
        <td></td>
                <td>
                  Key Path (internal use)
                </td>
                <td>
                  <input type="text" name="key_path" value="{}">
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
</div>
<div class="column small"></div>
</html>

"""



check_status_form = """<html>
{}
<body>
{}
<div class="column row"></div>
<div class="column left">
{}
</div>
<form action="/check_status?use_case={}&next_action={}" method="POST" enctype="multipart/form-data">

<div class="column middle">
  <div class="column middle_top">
  {}
  </div>

  <div class="column middle_middle">
  {}
  </div>

  <div class="column middle_bottom">
  <input type="submit" value="Check Status"></input>
  </div>
  </form>
</div>
<div class="column small">

</div>

</body>
</html>
"""

demonstrations_input ="""    <tr>
        <td></td>
        <td><b>Demonstrations </b>when enumerating the space. <br> Currently, only implemented for defect_detection.</td>
        <td><input type="file" name="demonstrations"></td>
        <td></td>
    </tr>
"""

separator_and_task_system_input = """    <tr>
        <td></td>
        <td><b>JSON key</b> for label. Should match the prompt.</td>
        <td><input width=70 type="text" name="label" value="{}"></input</td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td><b>Task System</b> Accompanies the prompt to be written.</td>
        <td><input type="text" name="task_system" rows=3 value="{}"></input></td>
        <td></td>
    </tr>

"""
email_and_password = """    <tr>
        <td></td>
        <td>Email Address</td>
        <td><input name="email_address" type="text"></input>
        <td></td>
    </tr>

    <tr>
        <td></td>
        <td>Password</td>
        <td><input name="password" type="text"></input>
        <td></td>
    </tr>
"""

enumerate_prompts =  """
<html>
{}
<body>
{}
<div class="column row"></div>
<div class="column small"></div>
<div class="column middle_big">
<form action="/enumerate_prompts?use_case={}" method="POST"  enctype="multipart/form-data">

 <div class="shaded">
 <table border=0>
    <tr>
        <td></td>
        <td colspan=2>
             <br>
             Here you design your Meta Prompt, the prompts that will write candidates for your ideal prompt.
             <br></td>
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
        <td><b>JSON key</b> for label. Should match the prompt.</td>
        <td><input width=70 type="text" name="label" value="{}"></input</td>
        <td></td>
    </tr>
    {}
    <tr>
        <td></td>
        <td><b>Evaluation </b>method for label. Should match the prompt output.</td>
        <td><select name="evaluator">
            <option value="accuracy">Accuracy</option>
            <option value="auc">AUC</option>
            </select>
        </td>
        <td></td>
    </tr>

    <tr>
        <td></td>
        <td><b>Model</b></td>
        <td>
            <table border=0>
                <tr>
                    <td>
                        <u>Model Name</u>
                    </td>
                    <td>
                        <u>N Prompts</u>
                    </td>
                    <td>
                        &nbsp; &nbsp; 
                    </td>
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
    {}
    <input type="hidden" name="batch_size" value="4"></input>
    <input type="hidden" name="n_batches" value="4096"></input>
    <tr>
        <td></td>
        <td></td>
        <td><input type="submit" name="submit" value="Submit"></input>
        <input type="submit" name="submit" value="Save"></input>
        <td></td>
    </tr>

 </table>
 </div>
</div>
<div class="column small"></div></form>
<div class="column row"></div>
"""
load_job = """

<html>
{}
<body>
{}
<div class="column row"></div>
<div class="column small"></div>
<div class="column middle_big">
<div class="column shaded">
<form action="/user_jobs" method="POST">

<table>
    <tr>
        <td></td>
        <td> ID </td>
        <td> <b>Setup Id </b></td>
        <td> User Prompt </td>
    </td>
    {}
    <tr>
      <td></td>
      <td></td>
      <td colspan=2><input type="submit" name="submit" value="Select"></input>
      </td></td>
    </tr>
</table>
</div>
{}
</form>
</div>
<div class="column small"></div>
<div class="column row"></div>
</html>

"""

load_prompt = """
<html>
{}
{}
<body>
<div class="column row"></div>
<div class="column small"></div>
<div class="column middle_big">
<div class="column shaded">
<form action="/user_library" method="POST">

<table>
    <tr>
        <td></td>
        <td> ID </td>
        <td> Prompt </td>
        <td> Date/Time </td>
        <td> <b>Use Case</b></td>
    </td>
    {}
    <tr>
      <td></td>
      <td></td>
      <td colspan=2><input type="submit" name="submit" value="Select"></input>
      </td></td>
    </tr>
</table>
</div>
{}
</form>
</div>
<div class="column small"></div>
<div class="column row"></div>
</html>
"""


navbar = """

<title>Promptimizer by Quante Carlo</title>
<div class="header">
    <p align="right">
    <table>
        <tr><td><h1>Promptimizer</h1></td>
            <td> &nbsp; &nbsp; &nbsp; &nbsp; </td><td rowspan=2>
            <img src="https://static.wixstatic.com/media/614008_6006e77a45db4c8ea97da77bc26cca7c~mv2.jpg/v1/fill/w_123,h_123,al_c,q_80,usm_0.66_1.00_0.01,enc_auto/qc%20logo.jpg"></img></p>
            </td>
        </tr>
        <tr><td align="right">by Quante Carlo</td><td> &nbsp; </td>
        </tr>
    </table>
    </p>
</div>



<div class="navbar">

    <a href="/">Home</a>
    <a href="/signup">Sign Up!</a>
    <a href='/buy_credits'>Buy Credits</a>
    <div class="subnav">
        <button class="subnavbtn">User Library</a><i class="fa fa-caret-down"></i></button>
        <div class="subnav-content">
            <a href="/load_prompt">Load Prompt</a>
            <a href="/load_job">Load Job</a>
        </div>
    </div>

    <div class="subnav">

        <button class="subnavbtn">Documentation</a><i class="fa fa-caret-down"></i></button>
        <div class="subnav-content">
            <a href="/how-it-works">How it works</a>
            <a href="/rag">How to prepare RAG</a>
            <a href="https://quantecarlo.com">Quante Carlo</a>
            <a href="/blog">Blog</a>
        </div>
    </div>
    <a href="/settings">Settings</a>
</div>


"""



header_and_nav = """<title>Promptimizer by Quante Carlo</title>
<div class="header">
<p align="right"><table><tr><td>
<h1>Promptimizer</h1></td>
<td> &nbsp; &nbsp; &nbsp; &nbsp; </td><td rowspan=2>
<img src="https://static.wixstatic.com/media/614008_6006e77a45db4c8ea97da77bc26cca7c~mv2.jpg/v1/fill/w_123,h_123,al_c,q_80,usm_0.66_1.00_0.01,enc_auto/qc%20logo.jpg"></img></p>
</td></tr>
<tr><td align="right">by Quante Carlo</td><td> &nbsp; </td> 
</tr></table>
</p>
</div>

<div class="topnav">
<a href="/">Home</a>
<a href="/">Signup</a>
<a href="https://quantecarlo.com">Quante Carlo</a>
<a href="/rag">How to prepare RAG</a>
<a href="/load_prompt">Load Prompt</a>

</div>
"""

sign_up = """<html>
{}
<body>
{}
<div class="column row"></div>
<div class="column small"></div>

<div class="column middle_big">
<br>

<div class="column shaded">
<font size="+1"><b>Sign Up</b></font>
<br>
<form action="/testuser" method="POST">
<table>
    <tr>
        <td>First Name</td>
        <td><input type="text" name="firstname" value=""></input></td>
    </tr>
    <tr>
        <td>Last Name</td>
        <td><input type="text" name="lastname" value=""></input></td>
    </tr>

    <tr>
        <td>Eail</td>
        <td><input type="text" name="email_address" value=""></input></td>
    </tr>
    <tr>
        <td>Password</td>
        <td><input type="text" name="password"></input></td>
    </tr>
    <tr>
      <td>
         <input type="submit" name="submit" value="submit"></input>
      </td>
    </tr>
</table>
</div>
</div>
<div class="column small"></div>
</form>

</html>
"""



sign_in = """<html>
{}
<body>
{}
<div class="column row"></div>
<div class="column small"></div>

<div class="column middle_big">
<br>

<div class="column shaded">
<font size="+1"><b>Sign In</b></font>
<br>
<form action="{}" method="POST">
<table>
    <tr>
        <td>Email</td>
        <td><input type="text" name="email_address" value=""></input></td>
    </tr>
    <tr>
        <td>Password</td>
        <td><input type="text" name="password"></input></td>
    </tr>
    <tr>
      <td>
         <input type="submit" name="submit" value="submit"></input>
      </td>
    </tr>
</table>
</div>
</div>
<div class="column small"></div>
</form>

</html>
"""


use_case_selector = """
<html>
{}
{}
<body>
<div class="column row"></div>
<table border=0>
    <tr>
        <td>
            &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
        </td>
        <td>
        </td>
        <td align='right'>
        </td>
        <td>
            &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
        </td>
    </tr>

        <tr>
            <td></td>
            <td>
                <h2>Tutorial Instructions</h2>
                <font size="+1">
                <ol><li>Decide on a Tutorial.</li>
                    <li>Download the corresponding training file.</li>
                    <li>Make a note of the corresponding metric.</li>
                    <li>Select the use case.</li>
                </ol>
            </td>
            <td>
                <b>Next Screens</b>
                <ol start="5">
                    <li>You will choose the corresponding metric on the next screen.</li>
                    <li>Click submit to create the search space</li>
                    <li>After the search space is created, you will upload a file of labeled data to perform the optimization.</li>
                </ol>
        </td>
        <td>
            &nbsp; &nbsp; &nbsp; &nbsp;
        </td>
    </tr>





    <tr><td colspan=4> &nbsp; </td></tr>
    <tr>
        <td>
            &nbsp; &nbsp; &nbsp; &nbsp;
        </td>
        <td colspan=2>
            <div class="shaded"><center>
            <table border=0>
                <tr>
                    <td>
                        <u>Use Case</u>
                    </td>
                    <td> &nbsp; </td>
                    <td>
                        <u>Description</u>
                    </td>
                    <td> &nbsp; </td>
                    <td>
                        <u>Source</u>
                    </td>
                    <td> &nbsp; </td>
                    <td>
                        <u>Prepared File</u>
                    </td>
                    <td> &nbsp; </td>
                    <td>
                        <u> Metric</u>
                    </td>
                </tr>
                <tr>
                    <td>
                        <a href="/prompt_preview?use_case=medical_diagnosis">Medical Diagnosis</a>
                    </td>
                    <td> &nbsp; </td>
                    <td>
                        Diagnose a patient based on text describing his or her symptoms.
                    </td>
                    <td> &nbsp; </td>
                    <td>
                        <a href="https://https://huggingface.co/datasets/gretelai/symptom_to_diagnosis">Hugging Face</a>
                    </td>
                    <td> &nbsp; </td>
                    <td>
                        <a href="data/medical_train_small.csv">medical_train_small.csv</a>
                    </td>
                    <td> &nbsp; </td>
                    <td>
                        Accuracy
                    </td>
                </tr>
                <tr>
                    <td>
                        <a href="/prompt_preview?use_case=ai_detector">AI Detector</a>
                    </td>
                    <td> &nbsp; </td>
                    <td>
                        Determine if the text was generated by a human or a language model.
                    </td>
                    <td> &nbsp; </td>
                    <td>

                        <a href="https://www.kaggle.com/competitions/llm-detect-ai-generated-text/data?select=train_essays.csv">Kaggle</a>
                    </td>
                    <td> &nbsp; </td>
                    <td>
                        <a href='/data/ai_generated.csv'>ai_generated.csv</a>
                    </td>
                    <td> &nbsp; </td>
                    <td>
                        AUC
                    </td>
                </tr>
                <tr>
                    <td>
                        <a href="/prompt_preview?use_case=defect_detector">Defect Detection</a>
                    </td>
                    <td> &nbsp; </td>
                    <td>
                       Determine if the part is defective from an image.
                    </td>
                    <td> &nbsp; </td>
                    <td>
                       <a href="https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product">Kaggle</a>
                    </td>
                    <td> &nbsp; </td>
                    <td>
                        <a href='/data/castings.csv'>castings.csv</a>
                    </td>
                    <td> &nbsp; </td>
                    <td>
                        AUC
                    </td>
                </tr>
            </table>
            </center>
            </div>
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
                <font size="+1">
                <b>Notes regarding Defect Detection Tutorial:</b>
                <ul>
                    <li>Defect Detection uses image data.</li>
                    <li>Defect Detection is the only use case that uses <i>demonstrations</i>. Demonstrations are few shot examples to choose from when making a prompt. The Defect Detection tutorial also optimizes the choice of image to use in the few shot examples.</li>
                    <li>On the next screen, there is a place to upload demonstrations. <b>Only upload demonstrations if selecting the Defect Detection use case.</b> </li>
                    <li>The tutorial for Defection Detection must use only open AI models to enumerate the search space.</li>
                </ul>
                <b>Notes when using your own prompts.</b><br>
                    &bull; The input file needs to have two columns labeled 'input' and 'output'.<br>
                    &bull; If you're using RAG, prepare the input file <a href="/rag">accordingly.</a><br>
                    &bull; There are three kinds of evaluators:
                    <ol><li>Accuracy - If target matches or not</li>
                        <li>AUC - probability must be from the following list: <i>'very unlikely', 'unlikely', 'equally likely and unlikely', 'likely', 'very likely'</i></li>
                        <li>AI prompt - there will be an additional prompt that evaluates the input and the answer and gives a 'correct' or 'incorrect' verdict.</li></ol>
               </font>
        </td>
        <td>
            &nbsp; &nbsp; &nbsp; &nbsp;
        </td>
    </tr>
</table>
</html>

"""


settings = """<htnl>
{}
{}
<body>
<h2>Settings</h2>
<div class="column small"></div>
<div class="column middle_big">
{}
</div>
<div class="column small"></div>
"""


