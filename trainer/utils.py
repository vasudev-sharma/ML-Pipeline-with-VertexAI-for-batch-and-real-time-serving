from google.cloud import storage
import yaml

def upload_model_gcs_bucket(model, bucketname, dest):
    """Uploads a file to Google Cloud storage

    Args:
        model (_type_): _description_
        bucketname (_type_): _description_
    """
    # TODO: Write full function
    client = storage.Client()
    bucket = client.bucket(bucketname)
    # blob 
    pass



def get_config_file(config_filepath):
    try:
        with open(config_filepath, 'r') as file:
            config = yaml.safe_load(file)
            return config
    except Exception as e:
        raise FileNotFoundError("File not found error")
    

