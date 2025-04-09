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
2. `pip-compile --upgrade -r requirements/requirements.in`
3. `pip install -r requirements/requirements.txt`

### Design Decision


# Add experiment Tracking with weights and biases: TODO later

# Data upload the dataset to Google Cloud bucket for reproducibility


# CI / CD:
Issue with CodeFresh artifact upload step




# Containerization: 
## Have 2 docker files
- Dev enviroment 
- prod evironment
<!--  -->
The image was build on MacOS with the following command: `docker buildx build --platform=linux/amd64 .`

if you want 

# Serving? 
<!-- How will we access the predcitions -->
<!-- What will be the considered the input to the model -->

## Local testing 
1. `docker build -f Dockerfile.production . -t inference_image`
2. `




# pip-compile



# Questions / thoughts



1. Why we can't create inference on our own container/image?
2. How will you package the data as well (training, validation and testing dataset)
3. Add instructions on how to test the endpoint?
4. How will you use CI/CD to automate the workflow?


Training question
1. How will you export model artifacts?


Serving questions (Decision)
1. Why we pin scikit -learn to 1.5.2 (issue with production image version)? 
2. Why I chose scikit-learn built image?
    "us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-6:latest"


## Predictions

JSON
{"instances": [[4.60042024e-03, 3.33172912e-01, 5.26700015e+00, 9.20000000e+01,0.00000000e+00, 0.00000000e+00, 2.80000000e+01, 4.00000000e+00,5.10000000e+01] ]}

```python
{'dist_to_restaurant': 0.004600420240541274, 'Hdist_to_restaurant': 0.3331729124701994, 'avg_Hdist_to_restaurants': 5.2670001515392695, 'date_day_number': 92.0, 'restaurant_id': 0.0, 'Five_Clusters_embedding': 0.0, 'h3_index': 28.0, 'date_hour_number': 4.0, 'restaurants_per_index': 51.0}
```


1. 


## Training
Issue with Vertex AI image (the base image has to be inherited from Scikit-learn image)