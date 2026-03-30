from groq import Groq as GroqClient
import time
from .Model import Model

class Groq(Model):
    def __init__(self, config):
        super().__init__(config)
        api_keys = config["api_key_info"]["api_keys"]
        api_pos = int(config["api_key_info"]["api_key_use"])
        self.api_key = api_keys[api_pos]
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        self.temperature = float(config["params"]["temperature"])
        self.client = GroqClient(api_key=self.api_key)

    def query(self, msg, image=None):
        trial = 0
        while trial < 20:
            try:
                return self.__do_query(msg)
            except Exception as e:
                print(f"  !! Groq query error (trial {trial}): {e}")
                trial += 1
                time.sleep(180)
        return ''

    def __do_query(self, msg):
        response = self.client.chat.completions.create(
            model=self.name,
            messages=[{'role': 'user', 'content': msg}],
            temperature=self.temperature,
            max_tokens=self.max_output_tokens
        )
        return response.choices[0].message.content
