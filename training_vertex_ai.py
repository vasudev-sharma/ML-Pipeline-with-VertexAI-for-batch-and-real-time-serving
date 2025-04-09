
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
    

    config = get_config_file("configs/training_config.yaml")
    # dataset = create_and_import_dataset_tabular_gcs_sample(display_name='data', project='keen-airlock-455922-q4', location='us-central1', gcs_source='gs://busyness-data/final_dataset.csv')
    aiplatform.init(project=config['PROJECT_ID'], location=config['REGION'], staging_bucket=config['data']['filename_uri'])

    job = aiplatform.CustomContainerTrainingJob(
                                            display_name=config['JOB_NAME'],
                                            command=["python", "-m", "trainer.training_script"],
                                            container_uri='us-central1-docker.pkg.dev/keen-airlock-455922-q4/skip-the-dishesv2/test_image:train', # TODO: Use custom container
                                            # container_uri='us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-6:latest', # TODO: Use custom container
                                            model_serving_container_image_uri=config["DEPLOY_IMAGE"], # 
                                        )
    MODEL_DISPLAY_NAME = "Testing model v2"

    if config['TRAIN_GPU']:

        model = job.run(
            model_display_name=MODEL_DISPLAY_NAME,
            replica_count=1,
             base_output_dir="gs://busyness-data",
            accelerator_type=config['GPU']['NVIDIA_TESLA_T4'],
            accelerator_count=config['GPU']['TRAIN_NGPU'],
            machine_type=config['TRAIN_COMPUTE'],
            # sync=True
        )

    else:

        model = job.run(
            model_display_name=MODEL_DISPLAY_NAME,
            base_output_dir="gs://vasu-bucket/artifacts",
            replica_count=1,
            machine_type=config['TRAIN_COMPUTE'], 
            # sync=True
        )




