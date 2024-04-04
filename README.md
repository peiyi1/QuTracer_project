# QuTracer_project

Running experiments requires an IBM quantum account. You can go to website https://quantum.ibm.com and create an account.

Before running experiments, save IBM account information by the following step: 

1. go to the path /QuTracer_project, open the file save_ibm_account_info.py, modify the arguments for the function IBMProvider.save_account based on your IBM account information. The information of token can be obtained in the Dashboard page from the website https://quantum.ibm.com; The information of instance can be obtained by checking your account plan, for example, if you are in open plan, then the instance can be set as 'ibm-q/open/main'
2. After complete the modification of the file save_ibm_account_info.py, run the file to save the IBM account information by the command:
   
   python save_ibm_account_info.py
