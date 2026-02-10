from dotenv import load_dotenv
import os
from infisical_sdk import InfisicalSDKClient

load_dotenv()

SECRET_HOST = os.environ.get("SECRET_HOST")
PROJECT_ID = os.environ.get("PROJECT_ID")
CLIENT_ID = os.environ.get("CLIENT_ID")
CLIENT_SECRET = os.environ.get("CLIENT_SECRET")
ENVIRONMENT = os.environ.get("ASYMPTOTIC_MODE")


def get_secrets():
    client = InfisicalSDKClient(host=SECRET_HOST)
    client.auth.universal_auth.login(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)

    secrets = client.secrets.list_secrets(
        project_id=PROJECT_ID, environment_slug=ENVIRONMENT, secret_path="/"
    ).secrets

    return {secret.secretKey: secret.secretValue for secret in secrets}


def get_secret(key, default=None):
    secrets = get_secrets()
    return secrets.get(key, default)
