# Prebuilts on Cerebrium

## Introduction
This repo consists of the prebuilt models for Cerebrium.   
Each folder in `./Prebuilts` contains a cortex deployment that can be used on Cerebrium. 
This consists of:
 - **config.yaml**: The configuration file for the cortex deployment. Contains information on the resources required for the deployment. 
 - **main.py**: The main file that is executed during the runtime
   - This has three parts
     - The `Item` which is the parameters for the API call
     - The `init` part which is only executed during the build time
     - The `predict` part which is executed during the runtime.
 - **requirements.txt**: The dependencies for the model
 - **pkglist.txt**: The list of apt packages that are installed during the build time

Additionally, it is adviced to place a small README.md to describe the model and its usage.

## How to use
The primary method to deploy prebuilt models is using the dashboard.  

Alternatively, you can use the Cortex CLI to deploy the model.   
In the folder containing the desired prebuilt model, run the following command:
```bash
cortex deploy <model_name> --config-file <config_file> 
```


## How to contribute
To contribute to this repo, please fork this repo and create a pull request.  
Additionally, please create an issue for any bugs or feature requests.

Don't hesitate to reach out to the team on slack or discord if there is anything you need help with or if you have any suggestions.