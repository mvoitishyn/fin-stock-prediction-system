from openai import OpenAI
import json

def get_assistant_response(api_key, assistant_id, message_content):
    client = OpenAI(api_key=api_key)

    thread = client.beta.threads.create()

    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=message_content
    )

    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant_id
    )

    if run.status == 'completed':
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        print("Assistant job is done.")
    else:
        print(run.status)
        return None

    def message_cleaned(message):
        message = messages.data[0].content[0].text.value
        message_cleaned = message.replace("```json", "").replace("\n", "").replace("```", "").replace("'", "")
        return message_cleaned
    
    def message_json(message):
        return json.loads(message)

    message  = message_cleaned(messages)
    
    try:
        return message_json(message)
    except:
        return message