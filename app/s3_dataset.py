import hashlib
import os
import tarfile
from pathlib import Path

import boto3
from botocore.config import Config

DATA_ROOT = Path(os.getenv("DATA_ROOT", "/data/cifar10"))

def download_dataset():
    bucket = os.environ["S3_BUCKET"]
    key = os.environ["S3_KEY"]
    expected_hash = os.environ["CIFAR10_SHA256"].lower()

    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    ready = DATA_ROOT / ".ready"
    dataset_dir = DATA_ROOT / "cifar-10-batches-py"

    if (
        ready.exists()
        and ready.read_text().strip() == expected_hash
        and (dataset_dir / "data_batch_1").exists()
        and (dataset_dir / "test_batch").exists()
    ):
        print("Dataset cache is ready", flush=True)
        return

    client = boto3.client(
        "s3",
        endpoint_url=os.environ["S3_ENDPOINT_URL"],
        region_name="us-east-1",
        config=Config(s3={"addressing_style": "path"}),
    )

    archive = DATA_ROOT / "cifar10.tar.gz"

    print(f"Downloading s3://{bucket}/{key}", flush=True)
    client.download_file(bucket, key, str(archive))

    digest = hashlib.sha256()
    with archive.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)

    if digest.hexdigest() != expected_hash:
        archive.unlink(missing_ok=True)
        raise RuntimeError("CIFAR-10 SHA256 mismatch")

    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            path = Path(member.name)

            if (
                not path.parts
                or path.parts[0] != "cifar-10-batches-py"
                or ".." in path.parts
                or path.is_absolute()
                or not (member.isfile() or member.isdir())
            ):
                raise RuntimeError(f"Unexpected archive entry: {member.name}")

        tar.extractall(DATA_ROOT, filter="data")

    if not (dataset_dir / "data_batch_1").exists():
        raise RuntimeError("CIFAR-10 extraction failed")

    ready.write_text(expected_hash)
    archive.unlink(missing_ok=True)

    print("CIFAR-10 downloaded and verified", flush=True)

if __name__ == "__main__":
    download_dataset()
