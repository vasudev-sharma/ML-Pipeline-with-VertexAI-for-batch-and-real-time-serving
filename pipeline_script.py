from typing import Union, List
import google.cloud.aiplatform as aiplatform
from src.utils import get_config_file
import argparse


def deploy_model(model, config):
    """Deploy a model to vertex AI endpoint"""

    endpoints = aiplatform.Endpoint.list(
        filter='display_name="{}"'.format(
            config["ENDPOINT_NAME"],
        )
    )

    # Check if endpoints exist, otherwise create an endpoint
    if len(endpoints) > 0:
        endpoint = endpoints[0]
    else:
        endpoint = aiplatform.Endpoint.create(
            display_name=config["ENDPOINT_NAME"],
            project=config["cloud"]["project_id"],
            location=config["cloud"]["region"],
        )

    # Deploy the model to vertex endpoint
    model.deploy(
        endpoint=endpoint,
        traffic_split={"0": 100},
        machine_type=config["hardware"]["deployment_machine_type"],
        min_replica_count=1,
        max_replica_count=1,
    )


def upload_model_to_vertex_registry(display_name, model_dir, deploy_image_path):
    """Upload a model to vertex AI registry"""
    model = aiplatform.Model.upload(
        display_name=display_name,
        artifact_uri=model_dir,
        serving_container_predict_route="/predict",
        serving_container_health_route="/health_check",
        serving_container_ports=[8080],
        serving_container_image_uri=deploy_image_path,
        is_default_version=True,
        version_aliases=["v1"],
        version_description="Busyness model",
    )

    return model


def create_and_import_dataset_tabular_gcs_sample(
    display_name: str,
    project: str,
    location: str,
    gcs_source: Union[str, List[str]],
):

    aiplatform.init(project=project, location=location)

    dataset = aiplatform.TabularDataset.create(
        display_name=display_name,
        gcs_source=gcs_source,
    )

    dataset.wait()

    print(f'\tDataset: "{dataset.display_name}"')
    print(f'\tname: "{dataset.resource_name}"')
    return dataset


def run_pipeline(args, config, pipeline_config):

    aiplatform.init(
        project=config["PROJECT_ID"],
        location=config["REGION"],
        staging_bucket=config["data"]["filename_uri"],
    )

    job = aiplatform.CustomContainerTrainingJob(
        display_name=config["JOB_NAME"],
        command=["python", "-m", "src.training_script"],
        container_uri=config["TRAIN_IMAGE"],
        model_serving_container_image_uri=config["DEPLOY_IMAGE"],  #
        model_serving_container_predict_route="/predict",
        model_serving_container_health_route="/health",
        model_serving_container_ports=[8080],
    )
    MODEL_DISPLAY_NAME = config["MODEL_DISPLAY_NAME"]

    # TODO: Save the dataset in Vertex AI
    # dataset = create_and_import_dataset_tabular_gcs_sample(display_name='data', project='keen-airlock-455922-q4', location='us-central1', gcs_source='gs://busyness-data/final_dataset.csv')

    if config["TRAIN_GPU"]:
        accelerator_count = config["GPU"]["TRAIN_NGPU"]
        accelerator_type = config["GPU"]["NVIDIA_TESLA_T4"]

    else:
        accelerator_count = 0
        accelerator_type = "ACCELERATOR_TYPE_UNSPECIFIED"

    model = job.run(
        model_display_name=MODEL_DISPLAY_NAME,
        base_output_dir=config["model"]["model_artifacts_uri"],
        replica_count=config["MAX_REPLICA"],
        accelerator_type=accelerator_type,
        accelerator_count=accelerator_count,
        machine_type=config["TRAIN_COMPUTE"],
        sync=config["SYNC"],
    )

    print("Model training completed successfully.")
    if model and args.batch:

        print("\n\n Generating Batch Predictions ................. \n\n")
        # Do batch predcitions with Vertex AI model
        batch_prediction_job = model.batch_predict(
            job_display_name=MODEL_DISPLAY_NAME + "-batch-job",
            gcs_source=f"gs://{config['data']['bucket_name']}/{config['data']['filename']}",
            gcs_destination_prefix=f"gs://{config['data']['bucket_name']}",
            instances_format="csv",
            machine_type=config["TRAIN_COMPUTE"],
            accelerator_count=accelerator_count,
            accelerator_type=accelerator_type,
            starting_replica_count=config["START_REPLICA"],
            max_replica_count=config["MAX_REPLICA"],
            sync=config["SYNC"],
        )

        batch_prediction_job.wait()
        print(batch_prediction_job.display_name)
        print(batch_prediction_job.resource_name)
        print(batch_prediction_job.state)

    # # DEPLOY ENDPOINT ONLINE

    # # Upload model
    # model = upload_model_to_vertex_registry(
    #     pipeline_config["model_name"],
    #     pipeline_config["MODEL_DIR"],
    #     pipeline_config["DEPLOY_IMAGE"],
    # )
    # print(model)

    if args.deploy:

        print(
            "\n\n Deploying Model to endpoint for online predictions ................. \n\n"
        )
        # Deploy model to endpoint
        deploy_model(model, pipeline_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether to run batch inference on model",
    )
    parser.add_argument(
        "--deploy",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether to deploy model to endpoint",
    )
    args = parser.parse_args()
    train_config = get_config_file("configs/training_config.yaml")
    pipeline_config = get_config_file("configs/pipeline_config.yaml")

    # Run Training + [optional] Batch inference + [optional]
    run_pipeline(args, config=train_config, pipeline_config=pipeline_config)
