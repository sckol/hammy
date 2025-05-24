from .simulator import Simulator
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

    def upload_simulator_results(self, simulator: Simulator):
        result_files = [
            simulator.calibration_results_cache_file,
            simulator.simulation_results_cache_file,
        ]
        for result_file in result_files:
            if result_file.exists():
                self.client.upload_file(
                    str(result_file),
                    self.bucket_name,
                    str(result_file.relative_to(simulator.RESULTS_DIR)),
                )
                print(
                    f"Uploaded {result_file} to Yandex Cloud Storage bucket {self.bucket_name}."
                )

    def download_simulator_results(self, simulator: Simulator):
        result_files = [
            simulator.calibration_results_cache_file,
            simulator.simulation_results_cache_file,
        ]
        for result_file in result_files:
            prefix = str(result_file.relative_to(simulator.RESULTS_DIR))
            if (
                next(
                    iter(
                        sorted(
                            (
                                x["Key"]
                                for x in self.client.list_objects_v2(
                                    Bucket=self.bucket_name, Prefix=prefix
                                ).get("Contents", [])
                            ),
                            reverse=True,
                        )
                    ),
                    None,
                )
                == prefix
            ):
                print(
                    self.client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
                )
                self.client.download_file(self.bucket_name, prefix, str(result_file))
                print(
                    f"Downloaded {result_file} from Yandex Cloud Storage bucket {self.bucket_name}."
                )
