import os
import json
from datetime import datetime
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
load_dotenv()

# API call logging
def log_api_call(call_type, model, input_data, response, error=None):
    """Log API calls to a JSON lines file for tracking and debugging."""
    log_dir = Path("api_logs")
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / f"api_calls_{datetime.now().strftime('%Y-%m-%d')}.jsonl"

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "call_type": call_type,
        "model": model,
        "input_preview": str(input_data)[:200] + "..." if len(str(input_data)) > 200 else str(input_data),
        "success": error is None
    }

    if error:
        log_entry["error"] = str(error)
    elif response:
        # Log response metadata
        if hasattr(response, 'usage'):
            log_entry["usage"] = {
                "prompt_tokens": getattr(response.usage, 'prompt_tokens', None),
                "completion_tokens": getattr(response.usage, 'completion_tokens', None),
                "total_tokens": getattr(response.usage, 'total_tokens', None)
            }
        if call_type == "completion" and hasattr(response, 'choices'):
            log_entry["response_preview"] = response.choices[0].message.content[:200] + "..."

    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')

def llm_completion(
    model, # String giving name of LLM
    temperature, # Temperature hyper-parameter,
    #system_prompt, # String giving system prompt
    user_prompt # String giving user prompt
    ):

    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()

    try:
        # All models, but Claude 3.7 Sonnet
        if not model in ['anthropic.claude-3-7-sonnet-20250219-v1:0']:

            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'),
                            base_url=os.getenv('OPENAI_BASE_URL'))

            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    #{"role": "developer", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                    ]
                )

        # Claude 3.7 Sonnet
        else:

            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'),
                            base_url=os.getenv('OPENAI_BASE_URL'))

            response = client.chat.completions.create(
                model=model,
                #temperature=temperature,
                messages=[
                    #{"role": "developer", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                    ],
                extra_body={
                        "thinking": { "type": "enabled", "budget_tokens": 2000 }
                    }
                )

        log_api_call("completion", model, user_prompt, response)
        return response

    except Exception as e:
        log_api_call("completion", model, user_prompt, None, error=e)
        raise

def llm_embedding(text):

    try:
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'),
                        base_url=os.getenv('OPENAI_BASE_URL'))

        response = client.embeddings.create(
            model="amazon.titan-embed-text-v2:0",
            input=text
            )

        log_api_call("embedding", "amazon.titan-embed-text-v2:0", text, response)
        return response.data[0].embedding

    except Exception as e:
        log_api_call("embedding", "amazon.titan-embed-text-v2:0", text, None, error=e)
        raise