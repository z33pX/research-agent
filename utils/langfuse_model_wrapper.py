from langfuse.model import TextPromptClient
from langfuse import Langfuse
from groq import Groq

import anthropic
import logging
import openai
import time

a = anthropic.Anthropic()
l = Langfuse()
g = Groq()


def langfuse_model_wrapper(
    name: str,
    system_prompt: str,
    user_prompt: str,
    prompt: TextPromptClient,
    model: str = "gpt-4o",
    temperature=0,
    host="openai",
    trace=None,
    observation_id=None,
):
    logging.info(f"Start inference '{name}' - model {model}, host {host}")
    if trace is None:
        trace = l.trace(name=name)
    if observation_id is None:
        if trace.id is not None:
            observation_id = trace.id

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    generation = trace.generation(
        name=name,
        model=model,
        input=messages,
        metadata={"temperature": temperature},
        prompt=prompt,
    )

    start = time.time()

    if host == "openai":
        completion = openai.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=messages,
        )
        result = completion.choices[0].message.content
        input_tokens = completion.usage.prompt_tokens
        output_tokens = completion.usage.completion_tokens
        total_tokens = completion.usage.total_tokens

    if host == "groq":
        completion = g.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=messages,
        )
        result = completion.choices[0].message.content
        input_tokens = completion.usage.prompt_tokens
        output_tokens = completion.usage.completion_tokens
        total_tokens = completion.usage.total_tokens
    if host == "anthropic":
        completion = a.messages.create(
            model=model,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt},
            ],
        )
        result = ""
        for component in completion.content:
            if component.type == "text":
                result += component.text

        input_tokens = completion.usage.input_tokens
        output_tokens = completion.usage.output_tokens
        total_tokens = input_tokens + output_tokens

    duration = time.time() - start

    generation.end(
        output={
            "result": result,
            "duration": duration,
        },
        usage={
            "input": input_tokens,
            "output": output_tokens,
            "total": total_tokens,
            "unit": "TOKENS",
        },
    )

    trace.score(
        name="ttps",
        value=total_tokens / duration,
        comment="The number of total tokens processed per second.",
        observation_id=observation_id,
    )
    trace.score(
        name="itps",
        value=input_tokens / duration,
        comment="The number of input tokens processed per second.",
        observation_id=observation_id,
    )
    trace.score(
        name="otps",
        value=output_tokens / duration,
        comment="The number of output tokens processed per second.",
        observation_id=observation_id,
    )

    return result
