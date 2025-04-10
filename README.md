
## Development Environment

We use pip-tools to manage dependencies. 
### Install / add Dependencies (`requirements.in`)
1. `pip install pip-tools`
2. `pip-compile --upgrade -r requirements/requirements.in`
3. `pip install -r requirements/requirements.txt`


## CI/CD Workflow
![CI/CD ](https://github.com/vasudev-sharma/skipTheDishes_ML_challenge/blob/main/images/ci_cd.png)

## Pipeline

![pipeline](https://github.com/vasudev-sharma/skipTheDishes_ML_challenge/blob/main/images/pipeline.png)

Entrypoint: `python pipeline_script.py`
The pipeline feeds in `configs/pipline_config.yaml` file to run the pipeline.


It offers two ways of running the pipeline:
1. With Batch Serving + Online 

## Training
Problem: To Predict the busyness of a certain geographical region. The project uses a dataset containing the courier locations captured during food collection at restaurants during a time interval. 


Training can be done in two modes: locally or on Cloud instance (such as Vertex AI)

1. To train locally, run `python -m src.training_script`
2. Otherwise, trigger the pipeline_script.py either on Cloud or CI/CD (with service account) to trigger training according



---

## Serving? 

### Batch Serving 

`python pipline_script.py --batch --no-deploy`

This will trigger src/trainer_script.py. On Vertex AI, we build at custom container (Dockerfile) with corresponding training dependencies to 


--- 
### Online Serving 

####  Build Inference image
1. `docker build -f Dockerfile.inference . -t inference_image` # NOTE: On Mac, use `docker buildx build --platform=linux/amd64 ...`
2. `docker run inference_image` # Inference server at 0.0.0.0:8080 gets started
3. Now, test predictions locally using CURL.



#### Testing
Use a post endpoint to thest the following predictions
```
{"instances": [[4.60042024e-03, 3.33172912e-01, 5.26700015e+00, 9.20000000e+01,0.00000000e+00, 0.00000000e+00, 2.80000000e+01, 4.00000000e+00,5.10000000e+01] ]}
```


