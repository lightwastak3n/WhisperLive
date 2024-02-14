
import os
from openai import OpenAI


class TranscriptProcessor:
    def __init__(self):
        self.openai_api_key = os.getenv("openai_api")
        self.models = {
                "gpt-4-0125-preview": {"context": 128000},
                "gpt-3.5-turbo-0125": {"context": 16385},
        }
        self.instructions = {
            "summarize": """You will be given a text that represents a lesson transcript.
        The main speaker does 90% of the talking. Sometimes others will ask questions and he will answer them.
        Please format everything in markdown. Create sections based on the topic that is being talked about.
        Be detailed, incorporate main ideas and essential information, eliminating extraneous language and focusing on critical aspects.
        Rely strictly on the provided text, without including external information.
        Take notes from the perspective of a listener. Use bullet points for main ideas. 
        Note should represent concrete advice that is being given.""",
            "fix_summary": """You will be given summary that is made out of multiple notes. 
        Please create one note out of it. Don't leave anything out unless it's already mentioned multiple times.
        Don't add any extra content.
        Fix the section names so that they don't repeat."""
        }

    def model_response(self, instructions: str, prompt: str, model="gpt-3.5-turbo-0125") -> str:
        """
        Prompts model with the given instructions and prompt and returns its response.

        Args:
            instructions (str): Instructions for the model either from self.instructions or provided by the user.
            prompt (str): Input text for the model.
            model (str): Name of the model. Defaults to gpt-3.5-turbo-0125 since better than gpt4 in price/performance

        Returns:
            str: Response from the model.
        """
        client = OpenAI(api_key=self.openai_api_key)

        completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": prompt}
                    ]
        )
        return completion.choices[0].message.content

    def summarize_transcript(self, transcript: str, model="gpt-3.5-turbo-0125") -> None:
        """
        Produces the summary for the given transcript. Based on the model size it summarizes it in chunks and then combines them.
        4 characters are roughly 1 token. We will leave 20% of context length for the summary.
        
        Args:
            transript (str): Text of the transcript.
            model (str): Name of the model to us: Name of the model to use.  

        """
        context_length = self.models[model]["context"] * 4
        if len(transcript) < context_length * 0.6:
            summary = self.model_response(self.instructions["summarize"], transcript, model)
        else:
            chunks = []
            sentences = transcript.split(".")
            current_chunk = []
            current_chunk_length = 0
            for sentence in sentences:
                # We accumulate characters until close to prompt limit (leave about 10-20% for summary)
                if current_chunk_length > context_length * 0.8:
                    chunks.append(".".join(current_chunk))
                    current_chunk = []
                    current_chunk_length = 0
                else:
                    current_chunk.append(sentence)
                    current_chunk_length += len(sentence)
            # Append the last chunk if it has content
            if current_chunk:
                chunks.append(".".join(current_chunk))

            for i, ch in enumerate(chunks):
                print(f"Chunk {i} has {len(ch)} characters.") 

            summaries = []
            for i, chunk in enumerate(chunks,1):
                print(f"Summarizing part {i}")
                part_summary = self.model_response(self.instructions["summarize"], chunk, model)
                summaries.append(part_summary)

            # TODO: What if this hits context limit (like a 10h transcript) 
            # Fixes summary since there might be some overlap in chunks summaries
            print(summaries)
            summary = self.model_response(self.instructions["fix_summary"], "\n".join(summaries), model)

        with open("summary.md", "w") as f:
            f.write(summary)

    def generate_text_name(self, transcript: str, head=1000) -> str:
        """
        Uses AI to generate name for the transcript based on the first head characters.
        
        Args:
            transcript (str): text of the transcript
            head (int): How many characters from the start of the text to take to generate name.

        Returns:
            str: Response from the model.

        """
        instructions = "You will be given part of a larger text, please provide the name that is 3-5 words long that reflects the text content."
        transcript_name = self.model_response(instructions, transcript[:head])
        transcript_name = transcript_name.replace('"', "").replace(" ", "_")
        return transcript_name
