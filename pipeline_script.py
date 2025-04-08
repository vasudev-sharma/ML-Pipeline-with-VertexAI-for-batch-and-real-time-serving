import google.cloud.aiplatform as aip
from trainer.utils import get_config_file
import os
from datetime import datetime




def deploy_model(model, config):
    """ Deploy a model to vertex AI endpoint
    """

    endpoints = aip.Endpoint.list(
        filter='display_name="{}"'.format(config['ENDPOINT_NAME'],
        order_by='create_time desc',
        project=config['cloud']['project_id'],
        location=config['cloud']['region'])
    )

    # Check if endpoints exist, otherwise create an endpoint
    if len(endpoints) > 0:
        endpoint = endpoints[0]
    else:
        endpoint = aip.Endpoint.create(
            display_name=config["ENDPOINT_NAME"],
            project=config['cloud']['project_id'],
            location=config['cloud']['region'])
    

    # Deploy the model to vertex endpoint
    model.deploy(endpoint=endpoint,
                 traffic_split={"0":100},
                 machine_type=config['hardware']['deployment_machine_type'],
                 min_replica_count=1,
                 max_replica_count=1
                 )

def upload_model_to_vertex_registry(display_name, model_dir, deploy_image_path):
    model = aip.Model.upload(
    display_name=display_name,
    artifact_uri=model_dir,
    serving_container_image_uri=deploy_image_path,
    is_default_version=True,
    version_aliases=["v1"],
    version_description="Busyness model",
                            )

    return model


if __name__ == '__main__':
    # Read config 
    pipeline_config = get_config_file("configs/pipeline_config.yaml")
    training_config = get_config_file("configs/pipeline_config.yaml")



    # Upload model
    model = upload_model_to_vertex_registry(pipeline_config['model_name'], pipeline_config['MODEL_DIR'], pipeline_config['DEPLOY_IMAGE'])
    print(model)


    # Deploy model to endpoint
    deploy_model(model, pipeline_config)

    


