"""
One-time AWS setup: creates S3 bucket, uploads data and model files, creates SQS queue.
Run once before starting the worker.

Usage:
    python aws/setup.py
"""

import os
import boto3
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent

UPLOAD_PATHS = [
    ("data/leads.json",             "data/leads.json"),
    ("data/call_logs.json",         "data/call_logs.json"),
    ("model/xgb_model.joblib",      "model/xgb_model.joblib"),
    ("model/label_encoders.joblib", "model/label_encoders.joblib"),
    ("model/feature_cols.joblib",   "model/feature_cols.joblib"),
    ("model/categories.joblib",     "model/categories.joblib"),
]


def create_s3_bucket(s3, bucket_name: str, region: str):
    print(f"Creating S3 bucket: {bucket_name}")
    try:
        if region == "us-east-1":
            s3.create_bucket(Bucket=bucket_name)
        else:
            s3.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={"LocationConstraint": region},
            )
        print(f"  Bucket created: s3://{bucket_name}")
    except s3.exceptions.BucketAlreadyOwnedByYou:
        print(f"  Bucket already exists, skipping")


def upload_files(s3, bucket_name: str):
    print("\nUploading files to S3:")
    for local_path, s3_key in UPLOAD_PATHS:
        full_path = PROJECT_ROOT / local_path
        s3.upload_file(str(full_path), bucket_name, s3_key)
        print(f"  Uploaded: {local_path} → s3://{bucket_name}/{s3_key}")


def create_sqs_queue(sqs, queue_name: str) -> str:
    print(f"\nCreating SQS queue: {queue_name}")
    response = sqs.create_queue(
        QueueName=queue_name,
        Attributes={"VisibilityTimeout": "60"},
    )
    queue_url = response["QueueUrl"]
    print(f"  Queue URL: {queue_url}")
    return queue_url


def main():
    region = os.environ["AWS_REGION"]
    bucket_name = os.environ["S3_BUCKET_NAME"]
    queue_name = os.environ["SQS_QUEUE_NAME"]

    s3 = boto3.client("s3", region_name=region)
    sqs = boto3.client("sqs", region_name=region)

    create_s3_bucket(s3, bucket_name, region)
    upload_files(s3, bucket_name)
    queue_url = create_sqs_queue(sqs, queue_name)

    print("\nSetup complete. Add this to your .env:")
    print(f"  SQS_QUEUE_URL={queue_url}")


if __name__ == "__main__":
    main()
