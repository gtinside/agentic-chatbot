# agentic-chatbot
The objective of this project is to develop a chatbot capable of interfacing with a RAG tool and automatically determining metadata filters. The focus is on functionality rather than a graphical user interface, making a command-line interface sufficient for interaction.

The chatbot will have the capability to access content from two provided documents. Additionally, it will preprocess queries to identify pertinent details about the source content to be included, such as which document and how much of it.

Classification logic plays a crucial role in the functionality of the chatbot. Before querying the documents, the chatbot will classify the nature of the query to determine which documents need to be included, filtering to include only those relevant to the context. Moreover, the bot will differentiate between specific questions and requests for summarization. For summarization requests, the entire document will be included in the context, while for specific questions, the chatbot will include contextually relevant snippets akin to typical RAG summaries.

To test the chatbot's performance, it will be run and its output saved for the following questions:
1. Summarize the "metagpt" paper.
2. Summarize the "autogen" paper.
3. Compare and contrast how "metagpt" and "autogen" handle roles.
4. What is conversation programming, and how does "autogen" utilize it?
5. What communication protocols are optimal for "metagpt"?


## Dependencies
- openai
- chromadb
- llama-index
- llama-index-embeddings-huggingface
- llama-index-vector-stores-chroma
- sentence-transformers
- pydantic
- langchain
- OPENAI_API_KEY

## Assets
- Documents are in ```assets/docs```
- Document metadata in ```assets/metadata.json```

## How to use it?
```python main.py --query "Summarize the metagpt paper"```

## Working
- Embeddings is generated and stored in an in-memory chromdb collection. llama-index is being used as the data framework to facilitate the interaction with chromadb
- User query is processed by an intent classifier that determines the document that needs to be queried and the type of query - Summarization or General Query
- The relevant context is then included and sent to OpenAI with the query, response is sent back to the user.
- For Summarization, all the documents of a given types are collated and set as context

## Sample Output
```
python main.py --query "Summarize the metagpt paper"                        
Document processed successfully
The query type is:  Intent.SUMMARIZATION
metagpt will be queried.
The MetaGPT paper introduces a novel meta-programming framework designed to enhance the problem-solving capabilities of multi-agent systems based on Large Language Models (LLMs). It leverages Standard Operating Procedures (SOPs) to streamline workflows, allowing agents with specialized roles (e.g., Product Manager, Architect, Engineer, etc.) to efficiently collaborate on complex tasks. MetaGPT employs structured communication and a publish-subscribe mechanism to manage information exchange among agents, reducing communication overhead and improving task execution efficiency.
```

```
python main.py --query "Summarize the autogen paper" 
Document processed successfully
The query type is:  Intent.SUMMARIZATION
autogen will be queried.
The AutoGen paper introduces an open-source framework designed to facilitate the development of applications using Large Language Models (LLMs) through multi-agent conversations. AutoGen enables developers to create customizable, conversable agents that can operate in various modes, employing combinations of LLMs, human inputs, and tools. The framework supports flexible conversation patterns and agent interaction behaviors, allowing for the development of complex LLM applications across various domains with reduced development effort. The paper showcases AutoGen's potential through six example applications
```

```
python main.py --query "What is conversation programming, and how does autogen utilize it?"   
Document processed successfully
The query type is:  None
autogen will be queried.
Conversation programming is a paradigm that focuses on computation and control flow within multi-agent conversations. AutoGen utilizes this by creating conversable agents with defined capabilities and roles, and programming their interaction behaviors through conversation-centric computation and control.
```

```
python main.py --query "What communication protocols are optimal for metagpt?"          
Document processed successfully
The query type is:  Intent.GENERAL
metagpt will be queried.
Communication protocols that enhance role communication efficiency, implement structured communication interfaces, and utilize an effective publish-subscribe mechanism.
```