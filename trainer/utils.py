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



def upload_to_gcs(bucket_name, source_file_path, destination_blob_name):
    """
    Uploads a file to a Google Cloud Storage bucket using default credentials.

    Args:
        bucket_name (str): Name of the GCS bucket.
        source_file_path (str): Path to the local file to upload.
        destination_blob_name (str): Desired name for the file in the bucket.
    """
    # Initialize the Google Cloud Storage client using default credentials
    storage_client = storage.Client()
    
    # Get the target bucket
    bucket = storage_client.bucket(bucket_name)
    
    # Create a blob object in the bucket
    blob = bucket.blob(destination_blob_name)
    
    # Upload the file to the blob
    blob.upload_from_filename(source_file_path)
    
    print(f"File {source_file_path} uploaded to gs://{bucket_name}/{destination_blob_name}")
    return f"gs://{bucket_name}/{destination_blob_name}"


def get_config_file(config_filepath):
    try:
        with open(config_filepath, 'r') as file:
            config = yaml.safe_load(file)
            return config
    except Exception as e:
        raise FileNotFoundError("File not found error")
    

