import openai
from langchain import PromptTemplate

class OpenAIRequests:
    template_str = """Use the following context to answer the question at the end.
                    If you don't know the answer, just say that you don't know, don't try to make up an answer.

                    Context:
                    -----------------
                    {context}
                    -----------------

                    Question: {query}
                    Short Answer:"""
    template = PromptTemplate(template=template_str, input_variables=["context", "query"])

    @staticmethod    
    def send_request(query, prompt):
        completion = openai.OpenAI().chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": query}
                ])

        return completion.choices[0].message.content  # Return the result of the query
    
    @staticmethod
    def send_request_with_context(context, query):
        qa_prompt =  OpenAIRequests.template.format(context=context, query=query)
        response = openai.OpenAI().chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "user", "content": qa_prompt}],
            temperature=0.0,
            max_tokens=100,
        )
        response.choices[0].message.content
        answer = response.choices[0].message.content
        return answer