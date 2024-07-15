from langfuse import Langfuse

import logging
import os
import re

l = Langfuse()


class Prompt:
    def __init__(self, prompt_id):
        self.prompt_id = prompt_id
        self.from_langfuse = True
        try:
            self.template = l.get_prompt(self.prompt_id)
        except Exception:
            logging.info(
                f"Loading prompt {prompt_id} from Langfuse failed. Loading from prompts.txt"
            )
            self.from_langfuse = False
            current_dir = os.path.dirname(os.path.realpath(__file__))
            prompt_folder = os.path.join(current_dir, "prompt_files")
            # Read all prompts.txt files. The file name is the prompt id.
            prompts = {}
            for file in os.listdir(prompt_folder):
                with open(os.path.join(prompt_folder, file), "r") as f:
                    prompts[file.split(".")[0]] = f.read()

            self.template = prompts.get(self.prompt_id)

            if self.template is None:
                raise ValueError(
                    f"Prompt with id {self.prompt_id} not found in folder {prompt_folder}"
                )

    def compile(self, **kwargs):
        template = self.template
        if not self.from_langfuse:
            template = re.sub(r"{{\s*(\w+)\s*}}", r"{\1}", template)
        return template.format(**kwargs)
