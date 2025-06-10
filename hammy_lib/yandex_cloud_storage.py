import os
import inspect
from .hammy_object import HammyObject
import boto3


class YandexCloudStorage:
    def __init__(self, access_key: str, secret_key: str, bucket_name: str):
        self.bucket_name = bucket_name
        self.client = boto3.session.Session(
            aws_access_key_id=access_key, aws_secret_access_key=secret_key
        ).client(
            service_name="s3",
            endpoint_url="https://storage.yandexcloud.net",
            region_name="ru-central1",
        )
        self.bucket_name = bucket_name

    def _get_hammy_objects_in_globals(self):
        # Walk up to the outermost (top) frame, but return to the original frame after
        frame = inspect.currentframe()
        try:
            top = frame
            while top.f_back is not None:
                top = top.f_back
            global_vars = top.f_globals
        finally:
            del frame
        return [obj for obj in global_vars.values() if isinstance(obj, HammyObject)]

    def _get_s3_key_from_object(self, obj):
        local_path = obj.filename
        results_dir = str(HammyObject.RESULTS_DIR).replace("\\", "/")
        s3_key = str(local_path).replace("\\", "/")
        if s3_key.startswith(results_dir + "/"):
            s3_key = s3_key[len(results_dir) + 1 :]
        return s3_key

    def upload_object(self, obj, overwrite: bool = False):        
        local_path = obj.filename
        if not os.path.exists(local_path):
            return
        s3_key = self._get_s3_key_from_object(obj)
        # Check if object exists on S3
        exists_on_s3 = False
        try:
            self.client.head_object(Bucket=self.bucket_name, Key=s3_key)
            exists_on_s3 = True
        except self.client.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                exists_on_s3 = False
            else:
                print(f"[S3] Error checking {s3_key}: {e}")
                return
        if exists_on_s3 and not overwrite:
            print(f"[S3] {s3_key} already exists. Skipping upload.")
            return
        self.client.upload_file(str(local_path), self.bucket_name, s3_key)
        print(f"[S3] Uploaded {s3_key} from {local_path}")

    def upload(self, overwrite: bool = False):
        for obj in self._get_hammy_objects_in_globals():
            self.upload_object(obj, overwrite=overwrite)

    def download_object(self, obj, overwrite: bool = False):
        local_path = obj.filename
        s3_key = self._get_s3_key_from_object(obj)
        # Check if file exists locally
        if os.path.exists(local_path) and not overwrite:
            print(f"[LOCAL] {local_path} already exists. Skipping download.")
            return
        # Check if object exists on S3
        try:
            self.client.head_object(Bucket=self.bucket_name, Key=s3_key)
        except self.client.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                print(f"[S3] {s3_key} does not exist on S3. Skipping download.")
                return
            else:
                print(f"[S3] Error checking {s3_key}: {e}")
                return
        # Download file
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        self.client.download_file(self.bucket_name, s3_key, str(local_path))
        print(f"[S3] Downloaded {s3_key} to {local_path}")

    def download(self, overwrite: bool = False):
        for obj in self._get_hammy_objects_in_globals():
            self.download_object(obj, overwrite=overwrite)
