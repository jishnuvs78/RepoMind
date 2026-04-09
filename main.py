import ingest
import search_agent 
import logs

import asyncio


REPO_OWNER = "huggingface"
REPO_NAME = "pytorch-image-models"

def initialize_index():
    print(f"Starting AI FAQ Assistant for {REPO_OWNER}/{REPO_NAME}")
    print("Initializing data ingestion...")

    pytorch_img_index, pytorch_img_vindex = ingest.index_data(REPO_OWNER, REPO_NAME, chunk=True)
    print("Data indexing completed successfully!")
    return pytorch_img_index, pytorch_img_vindex


def initialize_agent(pytorch_img_index, pytorch_img_vindex):
    print("Initializing search agent...")
    agent = search_agent.init_agent(pytorch_img_index, pytorch_img_vindex, REPO_OWNER, REPO_NAME)
    print("Agent initialized successfully!")
    return agent


def main():
    pytorch_img_index, pytorch_img_vindex = initialize_index()
    agent = initialize_agent(pytorch_img_index, pytorch_img_vindex)
    print("\nReady to answer your questions!")
    print("Type 'stop' to exit the program.\n")

    while True:
        question = input("Your question: ")
        if question.strip().lower() == 'stop':
            print("Goodbye!")
            break

        print("Processing your question...")
        response = asyncio.run(agent.run(user_prompt=question))
        logs.log_interaction_to_file(agent, response.new_messages())

        print("\nResponse:\n", response.output)
        print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    main()
