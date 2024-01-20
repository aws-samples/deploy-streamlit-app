import boto3
import json


class Llm:

    def __init__(self):
        # Create Bedrock client
        bedrock_client = boto3.client(
            'bedrock-runtime',
            # If Bedrock is not activated in us-east-1 in your account, set this value
            # accordingly
            region_name='us-east-1',
        )
        self.bedrock_client = bedrock_client

    def invoke(self, input_text):
        """
        Make a call to the foundation model through Bedrock
        """

        # Prepare a Bedrock API call to invoke a foundation model
        prompt = f"""\n\nHuman: {input_text}
                    \n\nAssistant:"""

        model_id = "anthropic.claude-v2"
        body = {
            "prompt": prompt,
            "max_tokens_to_sample": 4096,
            "temperature": 0.,
        }
        body = json.dumps(body)
        accept = 'application/json'
        contentType = 'application/json'

        # Make the API call to Bedrock
        response = self.bedrock_client.invoke_model(
            body=body, modelId=model_id, accept=accept, contentType=contentType
        )

        return response
