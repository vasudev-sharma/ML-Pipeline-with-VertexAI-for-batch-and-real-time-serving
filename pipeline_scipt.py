from google.cloud import aiplatform 
from utils import get_config_file
import os
from datetime import datetime



def deploy_model(model, config):
    """ Deploy a model to vertex AI endpoint
    """

    endpoints = aiplatform.Endpoint.list(
        filter='display_name="{}"'.format(config['ENDPOINT_NAME'],
        order_by='create_time desc',
        project=config['cloud']['project_id']
        location=config['cloud']['region'])
    )

    # Check if endpoints exist, otherwise create an endpoint
    if len(endpoints) > 0:
        endpoint = endpoints[0]
    else:
        endpoint = aiplatform.Endpoint.create(
            display_name=config["ENDPOINT_NAME"],
            project=config['cloud']['project_id']
            location=config['cloud']['region'])
    

    # Deploy the model to vertex endpoint
    model.deploy(endpoint=endpoint,
                 traffic_split={"0":100},
                 machine_type=config['hardware']['deployment_machine_type'],
                 min_replica_count=1,
                 max_replica_count=1
                 )

if __name__ == '__main__':
    # Read config 
    pipeline_config = get_config_file("configs/pipeline_config.yaml")
    

    # Upload model
    


