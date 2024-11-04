from openai import OpenAI
from groq import Groq
import anthropic
from google import generativeai as genai


class Model:
    def __init__(self, model_name):
        self.openai_client = OpenAI(api_key="<API-KEY>")
        self.groq_client = Groq(api_key="<API-KEY>")
        self.claude_client = anthropic.Anthropic(
            api_key="<API-KEY>")
        genai.configure(api_key="<API-KEY>")
        self.gemini_model = genai.GenerativeModel(model_name)
        self.model_name = model_name

    def __str__(self):
        return self.model_name

    def query(self, prompt, seed=9897, temperature=0):
        if self.model_name == "gpt-4":
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                seed=seed,
                temperature=temperature,
                max_tokens=50
            )
            return response.choices[0].message.content

        elif self.model_name == "gpt-4o":
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                seed=seed,
                temperature=temperature,
                max_tokens=50
            )
            return response.choices[0].message.content

        elif self.model_name == "llama3-70b":
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="llama3-70b-8192",
                seed=seed,
                temperature=temperature,
                max_tokens=50
            )
            return chat_completion.choices[0].message.content

        elif self.model_name == "llama3.1-70b":
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="llama-3.1-70b-versatile",
                seed=seed,
                temperature=temperature,
                max_tokens=50
            )
            return chat_completion.choices[0].message.content

        elif self.model_name == "claude-sonnet":
            message = self.claude_client.messages.create(
                model="claude-3-5-sonnet-latest",
                max_tokens=70,
                temperature=0.0,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            return message.content[0].text

        elif self.model_name.startswith("gemini"):
            response = self.gemini_model.generate_content(prompt)
            return response.text
