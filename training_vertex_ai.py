from typing import Union, List
import google.cloud.aiplatform as aiplatform
from trainer.utils import get_config_file


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


if __name__ == "__main__":

    # TODO: Add logging on Training and batch predictions

    config = get_config_file("configs/training_config.yaml")
    # dataset = create_and_import_dataset_tabular_gcs_sample(display_name='data', project='keen-airlock-455922-q4', location='us-central1', gcs_source='gs://busyness-data/final_dataset.csv')
    aiplatform.init(
        project=config["PROJECT_ID"],
        location=config["REGION"],
        staging_bucket=config["data"]["filename_uri"],
    )

    job = aiplatform.CustomContainerTrainingJob(
        display_name=config["JOB_NAME"],
        command=["python", "-m", "trainer.training_script"],
        container_uri=config["TRAIN_IMAGE"],  # TODO: Use custom container
        model_serving_container_image_uri=config["DEPLOY_IMAGE"],  #
    )
    MODEL_DISPLAY_NAME = config["MODEL_DISPLAY_NAME"]

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

    if model:
        print("Model training completed successfully.")

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

        # Add model deployment logic as well:
