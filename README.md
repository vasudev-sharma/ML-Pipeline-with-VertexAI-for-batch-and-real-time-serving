Problem description: The team was tasked with developing a model to predict the busyness of a certain geographical region. The project uses a dataset containing the courier locations captured during food collection at restaurants during a time interval. The data scientist has produced a working Proof of Concept (PoC). Now, as an ML Engineer, you are tasked with productionizing this PoC. This notebook contains the data scientist’s code to collect and create geo-location features to describe the busyness of regions (defined as h3 hexagons), and then train an ML model. The data scientist rushed to produce the PoC notebook, so the code is not well structured for a production application. As an ML engineer, your task is to:

Define a structured ML pipeline project.
Refactor the notebook into separate files to produce an executable ML pipeline using software engineering best practices and object-oriented programming appropriately.
Expected outcome: A structured ML pipeline project in a Git repo that you will talk us through, explaining your design choices. It should at least contain:

Scripts for each step
Training and prediction pipelines
Configurations file/s
Dependency management
CI/CD
Hints: We suggest containerization with Docker, using GCS for storage, Vertex AI to execute the pipeline, and GitHub Actions for CI/CD. But if you feel more comfortable with other tools that is ok.

Consider creating files for each step, for example, data_collection.py, feature_generation.py, training.py, and prediction.py, in addition to pipeline and config files to connect and execute the pipeline. Some features might be poorly implemented or not be in use. Your focus as an ML Engineer is refactoring the notebook into a structure project, but you can highlight any implementation issues you identify.


### Run locally with Python 3.9.7
1. `pip install pip-tools`
2. `pip-compile -r requirements/requirements.in`
3. `pip install -r requirements/requirements.txt`

### Design Decision


# Add experiment Tracking with weights and biases: TODO later



# Containerization
## Have 2 docker files
- Dev enviroment 
- prod evironment
<!--  -->
The image was build on MacOS with the following command: `docker buildx build --platform=linux/amd64 .`

if you want 

# Serving? 
<!-- How will we access the predcitions -->
<!-- What will be the considered the input to the model -->



# pip-compile
