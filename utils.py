import os
    #colored prints 
from termcolor import colored

def cosine_similarity_embeddings(embedding1, embedding2):
    return 1 - spatial.distance.cosine(embedding1, embedding2)


def delete_file_system_artifacts(directory):
    files_to_delete = ['.DS_Store', 'desktop.ini', 'Thumbs.db']

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file in files_to_delete:
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f'Successfully deleted: {file_path}')
                except OSError as e:
                    print(f'Error deleting {file_path}: {e}')
            
            


def handle_initial_user_inputs(research_question, openai_api_key):
    research_question = input("Provide the research question: ")
    model_mode = input("Provide the model mode ('1' for gpt-3.5-turbo, '2' for occasional use of gpt-4): ")
    if model_mode == "1":
        model_mode = "gpt-3.5-turbo"
    elif model_mode == "2":
        model_mode = "gpt-4"
    else:
        print("No valid model mode provided. Exiting.")
        exit()



    print(colored("Provide the search depth. The higher the search depth, the more results will be searched by the assistant. A depth of 10 may take up to 2 hours and cost about 4$ in API costs.", "yellow"))
    print(colored("This is an estimate based on two test runs. The actual costs may be higher or lower.", "red"))


    # User input
    search_depth = input("Provide the search depth (10 is default): ")
    try:
        if search_depth == "":
            search_depth = 10
        else:
            search_depth = int(search_depth)
    except:
        print("Please provide an integer value")
        search_depth = input("Provide the search depth (10 is default): ")
        if search_depth == "":
            search_depth = 11
        else:
            try:
                search_depth = int(search_depth)
            except ValueError:
                print("Again no valid integer. Exiting.")
                exit()
        

    return research_question, model_mode, search_depth