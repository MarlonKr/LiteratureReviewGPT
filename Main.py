import openai
import json
import os
from scipy import spatial
import time
import tiktoken
import PyPDF2
import os
from math import ceil
from fuzzywuzzy import process
from pathlib import Path
from config import OpenAIConfig
from termcolor import colored

openai.api_key = OpenAIConfig.OPENAI_API_KEY

BASE_DIR = Path(os.getcwd())

dirs = {
    "base": BASE_DIR,
    "prompts": BASE_DIR / "prompts",
    "resources": BASE_DIR / "resources",
    "resources_done": BASE_DIR / "resources" / "done",
    "embeddings": BASE_DIR / "embeddings",
    "embeddings_resources": BASE_DIR / "embeddings" / "resources",
    "embeddings_user": BASE_DIR / "embeddings" / "user",
    "core": BASE_DIR / "core",
    "core_trash": BASE_DIR / "core" / "trash",
    "core_paragraphs": BASE_DIR / "core" / "paragraphs",
    "core_paragraph_summaries": BASE_DIR / "core" / "paragraphs" / "summaries",
    "core_ConclusionAbstract": BASE_DIR / "core" / "ConclusionAbstract",
    "progress": BASE_DIR / "progress",
    "topic_evaluation": BASE_DIR / "topic_evaluation",
}

# Ensure all directories exist, if not, create them
for key, path in dirs.items():
    path.mkdir(parents=True, exist_ok=True)


def delete_ds_store_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == '.DS_Store':
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f'Successfully deleted: {file_path}')
                except OSError as e:
                    print(f'Error deleting {file_path}: {e}')

def is_json_serializable(obj):
    if isinstance(obj, str):
        try:
            json.loads(obj)
            return True
        except json.JSONDecodeError:
            return False
    else:
        try:
            json.dumps(obj)
            return True
        except (TypeError, json.JSONDecodeError):
            return False

def cosine_similarity_embeddings(embedding1, embedding2):
    return 1 - spatial.distance.cosine(embedding1, embedding2)

def get_embedding_string_jsondump(text, directory=str, filename=str, model="text-embedding-ada-002"):
    embedding = openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']
    with open(dirs["embeddings_user"] / f"{filename}.json", "w") as f:

        # dump the embedding and the text to a json file
        json.dump({"text": text, "embedding": embedding}, f)
    return embedding

def get_embedding_string(text, model="text-embedding-ada-002"):
    embedding = openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']
    
    return embedding

def truncate_messages(messages, encoder, max_tokens):
    truncated_messages = []
    current_tokens = 0
    for message in messages:
        content = message["content"]
        content_tokens = encoder.encode(content)
        current_tokens += len(content_tokens)

        if current_tokens > max_tokens:
            excess_tokens = current_tokens - max_tokens
            truncated_content = encoder.decode(content_tokens[:-excess_tokens])
            message["content"] = truncated_content
            current_tokens = max_tokens

        truncated_messages.append(message)

        if current_tokens == max_tokens:
            break

    return truncated_messages

def truncate_single_message(message, encoder, max_tokens):
    message_tokens = encoder.encode(message)

    if len(message_tokens) > max_tokens:
        truncated_message = encoder.decode(message_tokens[:max_tokens])
        return truncated_message
    else:
        return message

def send_message_to_chatgpt(message_input, role=None, model="gpt-3.5-turbo", temperature=0, include_beginning=True, is_list=False):

    encoder = tiktoken.encoding_for_model(model)
    max_tokens = 4050 if model == "gpt-3.5-turbo" else 8150 if model == "gpt-4" else None

    if not is_list:
        cleaned_message = message_input.replace("'", "").replace('"', '').replace("’", "")
        truncated_message = truncate_single_message(cleaned_message, encoder, max_tokens)
        
        message_input = [{"role": role, "content": truncated_message}]
    else:
        message_input = truncate_messages(message_input, encoder, max_tokens)

    final_message = beginning_message + message_input if include_beginning else message_input

    while True:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=final_message,
                temperature=temperature,
            )

            response_content = response.choices[0].message.content
            break  # If the API call is successful, break the loop

        except Exception as e:  # Catch any exception
            print(f"Error: {e}")
            print("Encountered an error. Retrying in 7 seconds.")
            time.sleep(7)  # Wait for 60 seconds before retrying


    print(colored('-'*50, 'yellow'))
    print(colored(f"message_input: ", 'cyan', attrs=['bold']) + colored(f"\n{final_message}", 'white'))
    print("\n\n")
    print(colored(f"response_content: ", 'green', attrs=['bold']) + colored(f"\n{response_content}", 'white'))
    print(colored('-'*50, 'yellow'))
    



    return response_content

def get_potential_topics(path_prompts, research_question):
    print(colored("getting potential topics", 'yellow', attrs=['bold']))
    
    # check if there is already a "potential_topics.json" file in the directory topic_evaluation
    if (dirs["topic_evaluation"] / "potential_topics.json").exists():
        with open((dirs["topic_evaluation"] / "potential_topics.json"), "r") as f:
            potential_topics = json.load(f)

    else:
            
        with open(path_prompts / "example1.txt", "r") as f:
            task_list_prompt_1 = f.read()

        with open(path_prompts / "example2.txt", "r") as f:
            task_list_prompt_2 = f.read()

        potential_topics = [
            {"role": "user","content": "Provide an extensive, flat, bulleted list of potential headings, arguments, topics for an outline of a systematic review paper on the research question provided by the user. Don't write the final outline yet! Make sure that you only provide a comprehensive pool of suitable headings, arguments, topics that would fit into the outline of a paper on the research question. The assignment paper is a systematic review and is designed to examine the given resources in order to provide an answer to the research question. It won't produce any data itself, it will rely entirely on the data provided by the resources and draw conclusions from them. As mentioned before, don't make an outline yet, just provide an unordered list with no structure that can be used to make the outline later. Skip main headings such as 'Introduction', 'Methodology' and so on. Just provide a linear bulleted list, nothing else! Acknowledge by typing 'ok.'"},
            {"role": "assistant","content": "ok."},
            {"role": "user", "content": "Research question: Does the use of an ensemble of decision trees (such as Random Forest) consistently outperform a single decision tree in solving regression problems?"},
            {"role": "assistant","content": task_list_prompt_1},
            {"role": "user", "content": "Research question: Has China's growing income inequality hindered its economic development and stability?"},
            {"role": "assistant","content": task_list_prompt_2},
            # for testing purposes:
            #{"role": "user", "content": "Research question: " + research_question + ". This time only provide five possible topics and therefore only two bullet points. Nothing more."},
            {"role": "user", "content": "Research question: " + research_question},
        ]

        potential_topics = send_message_to_chatgpt(message_input=potential_topics, role="user", model=model_mode, temperature=0.7, include_beginning=True, is_list=True)

        return potential_topics

def eval_potential_topics(potential_topics, status, research_question):
    print(colored("evaluating potential topics", 'yellow', attrs=['bold']))
    for topic, topic_info in potential_topics.items():
        current_topic_name = topic
        current_topic_description = topic_info["description"]
        if topic_info["status"] == "unchecked":
            # update status to "in progress"
            # check if there is already a directory for the current topic
            if not (dirs["topic_evaluation"] / current_topic_name).exists():
                (dirs["topic_evaluation"] / current_topic_name).mkdir()
                print(colored(f"created directory for topic {current_topic_name}", 'green', attrs=['bold']))
            else:
                print(colored(f"directory for topic {current_topic_name} already exists", 'green', attrs=['bold']))
        
        if topic_info["status"] == status or status == "all":
            print(colored(f"evaluating topic {current_topic_name}, status {topic_info['status']}", 'yellow', attrs=['bold']))
            input_embedding = get_embedding_string_jsondump(text=current_topic_name + " " + current_topic_description, directory=dirs["embeddings_user"], filename=current_topic_name)
                # calculate cosine similarity between input and all embeddings in path_embeddings_resources
            similarity_dict = {}
            for file in dirs["embeddings_resources"].glob("*.json"):
                with file.open("r") as f:
                    # do something with the file contents here

                    data = json.load(f)
                    # get embedding from json
                    embedding = data["embedding"]
                    # get text from json
                    text = data["text"]
                    page = data["page"]
                    # calculate cosine similarity
                    similarity = cosine_similarity_embeddings(input_embedding, embedding)
                    # extract the relevant information from the text
                    
                    # add to similarity_dict
                    similarity_dict[similarity] = {"text": text, "page": page, "filename": file}
            # sort similarity_dict
            similarity_dict = dict(sorted(similarity_dict.items(), reverse=True))
            # get top 13 results and save each text in a json file
            i = 0
            text_list ={}
            def split_text_into_chunks(text, max_tokens):
            
                enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
                tokens = enc.encode(text)
                num_tokens = len(tokens)
                num_chunks = ceil(num_tokens / max_tokens)

                chunks = []
                for i in range(num_chunks):
                    start = i * max_tokens
                    end = (i + 1) * max_tokens
                    chunk = enc.decode(tokens[start:end])
                    chunks.append(chunk)

                return chunks
            
            # iterate over the similarity dict and get the text and page
            for similarity, text_page_file in similarity_dict.items():            
                if i < search_depth :
                    text = text_page_file["text"]
                    page = text_page_file["page"]
                    file = text_page_file["filename"]
                    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
                    tokens = enc.encode(f"Quote everything from the text above that is relevant to {current_topic_name}.")
                    max_tokens = 4080 - len(tokens)
                    text_chunks = split_text_into_chunks(text, max_tokens)

                    for chunk in text_chunks:
                        token_chunk = enc.encode(chunk)

                        response = send_message_to_chatgpt(
                            message_input=f"INPUT: {chunk}. TASK: Quote everything from the text above that is relevant to the topic: {current_topic_name}.",
                            role="user",
                            model="gpt-3.5-turbo",
                            temperature=0,
                            is_list=False,
                            include_beginning=True,
                        )
                        print(f"Quotes: {response}")
                        # append response and page to text_list dict without using index
                        text_list[i] = {"topic": current_topic_name, "description": current_topic_description ,"text": chunk, "quotes": response, "page": page, "filename": str(file)}
                        
                    i = i + 1

            # save text_list to json file
            with (dirs["topic_evaluation"] / current_topic_name / "sources.json").open("w") as f:
                json.dump(text_list, f)


            # for each text in text_list, create a bullet point summary 
            bullet_points = []
            for key in text_list:
                topic = text_list[key]["topic"]
                description = text_list[key]["description"]
                text = text_list[key]["text"]
                quotes = text_list[key]["quotes"]
                page = text_list[key]["page"]
                filename = text_list[key]["filename"]
                # create a bullet point summary
                bullet_point_summary = send_message_to_chatgpt(message_input=f"{text}\n Source:{filename}, page {page}--- \n Create a brief bullet point summary of the text above. Name the source and the page in one short line. Only respond with the bullet point summary and the source, nothing else.", model="gpt-3.5-turbo", role= "user", temperature=0, is_list=False, include_beginning=True)
                # append bullet point summary to text
                print(f"bullet point summary: {bullet_point_summary}")
                bullet_points.append(bullet_point_summary)

            # save bullet points to json file
            with (dirs["topic_evaluation"] / current_topic_name / "Relevant_Text_Bullet_Points.json").open("w") as f:
                json.dump(bullet_points, f)

            
            message_temp = [
                {"role": "system", "content": f"You are a topic evaluator. You will receive a research question, a topic name and a description. You acknowledge this information with 'acknowledged'. Then the user asks you to evaluate whether the information the user provides is sufficient in order to include the topic into the outline of a systematic review paper on the respective research question'. If yes, you will answer with 'sufficient. Provided info is complete enough to include it into the outline of the paper'. If not, you will answer with 'Insufficient, source info lacks from relevant parts'."},
                {"role": "user", "content": f"TOPIC NAME: {current_topic_name}. TOPIC DESCRIPTION: {current_topic_description}. RESEARCH QUESTION: {research_question}. SOURCE: {filename} Acknowledge by saying 'acknowledged'."},
                {"role": "user", "content": f"SOURCE INFO: {bullet_points}. Acknowledge again by saying 'acknowledged'."},
                {"role": "assistant", "content": "acknowledged"},
                {"role": "user", "content": f"COMMAND: Evaluate whether the provided summary of the source info and the diversity of the sources indicate that the source info is sufficient for writing about the TOPIC NAME as a part of an assignment paper on the RESEARCH QUESTION and therefore including this topic into the outline of my assignment paper as one of the outline points. If yes, you will answer with 'sufficient. Provided info is complete enough to include it into the outline of the paper'. If not, you will answer with 'Insufficient, source info lacks from relevant parts'. Only answer with one of the two options, nothing else."},
            ]
            enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
            tokens = enc.encode(str(message_temp))
            num_tokens = len(tokens)
            while num_tokens > 4080:
                # remove first element from message_temp
                if "SOURCE INFO" in message_temp[0]["content"]:
                    # skip removing the first element
                    # count tokens of the first element
                    bullet_points_tokens = enc.encode(message_temp[0]["content"])
                    bullet_points_tokens_num = len(bullet_points_tokens)
                    rest_tokens_num = num_tokens - bullet_points_tokens_num
                    num_tokens = 4080 - rest_tokens_num
                    if bullet_points_tokens_num > num_tokens:
                        message_temp[0]["content"] = enc.decode(enc.encode(message_temp[0]["content"])[:num_tokens])
                    break
                else:
                    message_temp.pop(0)
                    # recalculate num_tokens
                    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
                    tokens = enc.encode(str(message_temp))
                    num_tokens = len(tokens)
                
            evaluation = send_message_to_chatgpt(message_input=message_temp, model="gpt-3.5-turbo", temperature=0.1, include_beginning=True, is_list=True, role="user")
            print(current_topic_name + ": " + evaluation)  
            if "insufficient" in evaluation.lower():
                sufficiency = "insufficient"
            elif "sufficient" in evaluation.lower():
                sufficiency = "sufficient"
            else:
                sufficiency = "no answer"

            # update the status of the topic in the json file
            potential_topics[current_topic_name]["status"] = sufficiency

            potential_topics[topic]["status"] = sufficiency # do I need to save this? answer: yes, because I need to save the status of the topic. how do I save it? answer: you save it in the json file of the topic by overwriting the json file with the new status. Like this:
            
            with (dirs["topic_evaluation"] / current_topic_name / f"{current_topic_name}_EVAL.json").open("w") as f:
                json.dump(potential_topics[topic], f)


        else:

            print(f"Status of {current_topic_name} is {topic_info['status']}, not {status}.")
    
    # update potential_topics to json file
    with (dirs["topic_evaluation"] / "potential_topics.json").open("w") as f:
        json.dump(potential_topics, f)

        
    # iterate over all files and subdirectories in dir_topic_evaluation and look for "_EVAL" filenames. If there is one, check the status. If the status is "sufficient", add the topic to the list of topics that are sufficient. 

    sufficient_topics = []
    insufficient_topics = []
    for dirpath, dirnames, files in os.walk(str(dirs["topic_evaluation"])):
        for file in files:
            if file.endswith("_EVAL.json"):
                file_path = os.path.join(dirpath, file)
                with open(file_path, "r") as f:
                    data = json.load(f)
                    if data["status"] == "sufficient":
                        #sufficient_topics.append(data["topic"]) ERROR: data["topic"] does not exist. I need to get the topic name from the filename.
                        # name is without "_EVAL.json"
                        topic_name = file[:-9]
                        sufficient_topics.append(topic_name)
                    if data["status"] == "insufficient":
                        #insufficient_topics.append(data["topic"])
                        topic_name = file[:-9]
                        insufficient_topics.append(topic_name)
    
    return sufficient_topics, insufficient_topics
       
def conduct_research(input, path_embeddings_resources, extract_relevancy_to ="", main_instruction =""):

    # get embedding for input
    input_embedding = get_embedding_string_jsondump(input, str(dirs["embeddings_user"]), "input")
    # calculate cosine similarity between input and all embeddings in path_embeddings_resources
    similarity_dict = {}
    for file in os.listdir(path_embeddings_resources):
        if file.endswith(".json"):
            with open(f"{path_embeddings_resources}{file}", "r") as f:
                data = json.load(f)
                # get embedding from json
                embedding = data["embedding"]
                # get text from json
                text = data["text"]
                # calculate cosine similarity
                similarity = cosine_similarity_embeddings(input_embedding, embedding)
                # extract the relevant information from the text
                  
                # add to similarity_dict
                similarity_dict[similarity] = text
    # sort similarity_dict
    similarity_dict = dict(sorted(similarity_dict.items(), reverse=True))
    # get top 13 results and save each text in a json file
    i = 0
    text_list = []
    for similarity, text in similarity_dict.items():
        if i < 10:
            if extract_relevancy_to != "":
                text = send_message_to_chatgpt(message_input=f"{text}\n --- \n Quote everything from the text above that is relevant to '{extract_relevancy_to}.", model="gpt-3.5-turbo", temperature=0, include_beginning=True, is_list=False, role="user")
            else:
                pass

            # append text to list
            text_list.append(text)

        i = i + 1
    
    return text_list
          

    pass

def embed_resources(input_path=dirs["resources"], output_path=dirs["embeddings_resources"]):
    # walk the input_path and get all pdf files
    for file in os.listdir(input_path):
        print(file)
        if file.endswith(".pdf"):
            print(f"\n\nprocessing {file}")
            with open(input_path / file, "rb") as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                # Get the number of pages in the PDF
                num_pages = len(pdf_reader.pages)

                # Loop through each page and extract text
                for page_num in range(num_pages):
                    # Get the page object
                    page = pdf_reader.pages[page_num]
                    
                    # Extract text from the page
                    text = page.extract_text()

                    # Get the embedding for the text
                    embedding = get_embedding_string(text=text)
                    #"embedding created for page" in yellow, "page number" in green
                    print(colored(f"embedding created for {file}, page ", "yellow"), colored(f"{page_num}", "green"))

                    # Save the embedding, text, and page number in a json file
                    with open(f"{output_path}/{file}_{page_num}.json", "w") as f:
                        json.dump({"embedding": embedding, "text": text, "page": page_num}, f)
            # move the pdf file to the dir_resources/done folder. Create the folder if it does not exist
            if not os.path.exists(f"{input_path}/done"):
                os.makedirs(f"{input_path}/done")
            # dont use shutil because it is not working
            os.rename(f"{input_path}/{file}", f"{input_path}/done/{file}")

        else:
            pass

    return "done"


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
    

if "/apikey" in research_question.lower() or not OpenAIConfig.OPENAI_API_KEY:
    print("You need to provide an API key.")
    OpenAIConfig.OPENAI_API_KEY = input("API key: ")


beginning_message = [
    {"role": "system", "content": f"You are an assistant for writing a term paper. You will help the user write a systematic review paper for Harvard University. The research question of the paper is {research_question}. The user will give you various tasks from different stages of the work process. Follow the user's instructions carefully, think step-by-step before answering, and generally stick to scientific and precise language. Always respond with the requested output, never elaborate on your reasoning, and never say anything at the meta-level. You are not allowed to ask questions. You only provide the required output, nothing else."},
    {"role": "assistant", "content": "Acknowledged. What is my current task?"},
    ]

delete_ds_store_files(BASE_DIR)

embedding = embed_resources(input_path=Path(dirs["resources"]), output_path=Path(dirs["embeddings_resources"]))
if embedding == "done":
    print(colored("embeddings created", "green"))
else:
    print(colored("embeddings failed, retry again", "red"))
    exit()

potential_topics = get_potential_topics(dirs["prompts"], research_question=research_question)

if is_json_serializable(potential_topics):
    print("found potential topics .json\n")
    pass

else:
    print("creating potential topics .json\n")
    print("Potential topics:\n")
    print(potential_topics)
    topic_dict = {}

    for line in potential_topics.splitlines():

        # skip empty lines
        if not line.strip():
            continue

        # create the system and user messages
        message_temp = f"Potential topic: {line}. Command: Write a brief description of this potential topic. Also, create a checklist with the necessary information to write a reasonable paragraph about this topic. The checklist must be written in relation to the research question that the paper will answer."

        # generate the task description using ChatGPT

        task_description = send_message_to_chatgpt(message_input=message_temp,model= "gpt-3.5-turbo", role="user", temperature=0.7, include_beginning=True, is_list=False)

        # add the topic and its description to the dictionary
        topic_dict[line] = {"description": task_description, "status": "unchecked"}

    # write the dictionary to the JSON file
    with open(str(dirs["topic_evaluation"] / "potential_topics.json"), "w") as f:
        json.dump(topic_dict, f)


with open(str(dirs["topic_evaluation"] / "potential_topics.json"), "r") as f:
    potential_topics = json.load(f)
        
evaluated_topics, insufficient_topics = eval_potential_topics(potential_topics=potential_topics, status="unchecked", research_question=research_question)


print(colored("\n\nEvaluated topics:\n", "yellow", attrs=["bold"]))
print(colored(evaluated_topics, 'green'))

print(colored("\n\nInsufficient topics:\n", "yellow", attrs=["bold"]))
print(colored(insufficient_topics, 'red'))

if (dirs["core"] / "outline.txt").exists():
    outline = (dirs["core"] / "outline.txt").read_text()
else:
    message_outline = f"Evaluated topics:\n {str(evaluated_topics)}\n\n Task: Choose appropriate topics from the pool of provided topics and create an outline with them for an systemical review assignment paper that tries to answer the research question '{research_question}'. Your task is to make a thoughtful choice that aims to choose and well-fitting topics that chained altogether make a sufficent outline for a harvard assignment paper on the research question. Only respond with the outline, nothing else."
    outline = send_message_to_chatgpt(message_input=message_outline, role="user", model=model_mode, temperature=0.7, include_beginning=False, is_list=False)
    with open((dirs["core"] / "outline.txt"), "w") as f:
        f.write(outline)
print(colored("Outline:", "yellow"))
print(outline)


# for each evaluated topic, send chatgpt a message to create a paragraph
interim_paragraph= ""

# if there is a file that ends with txt in the dir_core/paragraphs folder, then pass


check = [f for f in os.listdir(dirs["core_paragraphs"]) if os.path.isfile(os.path.join(dirs["core_paragraphs"], f))]
if len(check) > 0:
    print(len(check))
    print(check)
    print("paragraphs already exist")
    pass

else:
    print("creating paragraphs")
    for topic in evaluated_topics:
        print(topic)
        #def remove_special_chars(string):
            #return re.sub(r'[0-9\._\W]+', '', string)
        
        #topic = remove_special_chars(topic)
        # list directory topic_evaluation and check if topic is part of a folder name
        found_folder = False
        for file in os.listdir(dirs["topic_evaluation"]):
            if os.path.isdir(os.path.join(dirs["topic_evaluation"], file)):

                if topic in file:
                    # get the topic description from the file
                    with open(dirs["topic_evaluation"].joinpath(file, "sources.json"), "r") as f:
                        sources = json.load(f)
                        print(sources)
                        found_folder = True

                    break
        if found_folder == False:
            print("using fuzzywuzzy")
            def find_most_similar_file(dir_topic_evaluation, topic):
                files = os.listdir(dir_topic_evaluation)
                most_similar_file, similarity = process.extractOne(topic, files)
                return most_similar_file
            
            most_similar_file = find_most_similar_file(dirs["topic_evaluation"], topic)
            with open(dirs["topic_evaluation"] / most_similar_file / "sources.json", "r") as f:
                sources = json.load(f)
                print(sources)
                found_folder = True

            
        for source in sources.values():
            filename = source["filename"]
            page = source["page"]
            quote = source["quotes"]
            topic_description = source["description"]

            message = f"Topic: {topic}. Source: {filename}, page {page}.\n\n Content: {quote}.\n\n Command: Write a section about the topic using the content of the source provided. Provide citation by naming the source and the page. Use the checklist to ensure that you have included all necessary information in the paragraph: {topic_description}. \n\n This is the current state of the paragraph: {interim_paragraph}. Modify the paragraph by adding or incorporating new information or removing redundant information. Make sure to handle the already existing citations properly and in the correct order. You may conclude that the content of the source is not relevant or too redundant (e.g. no new relevant information available) for the paragraph. In this case, you may respond with: 'Nothing to add, skip'. Never respond with anything else than the new version of the paragraph or 'Nothing to add, skip'."

            # count tokens
            enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
            tokens = enc.encode(message)
            tokens_len = len(tokens)
            if tokens_len > 4090:
                message = f"Topic: {topic}. Source: {filename}, page {page}.\n\n Content: {quote}.\n\n Command: Write a section about the topic using the content of the source provided. Provide citation by naming the source and the page. Use the checklist to ensure that you have included all necessary information in the paragraph: {topic_description}. \n\n This is the current state of the paragraph: {interim_paragraph}. Modify the paragraph by adding or incorporating new information or removing redundant information. Make sure to handle the already existing citations properly and in the correct order. You may conclude that the content of the source is not relevant or too redundant (e.g. no new relevant information available) for the paragraph. In this case, you may respond with: 'Nothing to add, skip'. Never respond with anything else than the new version of the paragraph or 'Nothing to add, skip'."
                tokens = enc.encode(message)
                tokens_len = len(tokens)
                if tokens_len > 4090:
                    message = f"Topic: {topic}. Source: {filename}, page {page}.\n\n Content: {quote}.\n\n Command: Write a section about the topic using the content of the source provided. Provide citation by naming the source and the page. \n\n Use the checklist to ensure that you have included all necessary information in the paragraph: {topic_description}. \n\n This is the current state of the paragraph: {interim_paragraph}. Modify the paragraph by adding or incorporating new information or removing redundant information. Make sure to handle the already existing citations properly and in the correct order. You may conclude that the content of the source is not relevant or too redundant (e.g. no new relevant information available) for the paragraph. In this case, you may respond with: 'Nothing to add, skip'. Never respond with anything else than the new version of the paragraph or 'Nothing to add, skip'."
                    tokens = enc.encode(message)
                    tokens_len = len(tokens)
                    if tokens_len > 4090:
                        message_tokens = enc.encode(message)
                        message_tokens_len = len(message_tokens)
                        quote_tokens = enc.encode(quote)
                        quote_tokens_len = len(quote_tokens)
                        available_tokens = message_tokens_len - quote_tokens_len
                        available_tokens = 4090 - available_tokens
                        quote = enc.decode(enc.encode(quote)[:available_tokens])
                        message = f"TOPIC: {topic}. SOURCE: {filename}, PAGE {page}.\n\n CONTENT: {quote}.\n\n Command: Write a section about the TOPIC using the CONTENT of the SOURCE provided. Provide citation by naming the SOURCE and the PAGE. Remember that \n\n This is the current state of the paragraph: {interim_paragraph}. Modify the paragraph by adding or incorporating new information or removing redundant information. Make sure to handle the already existing citations properly and in the correct order. You may conclude that the content of the source is not relevant or too redundant (e.g. no new relevant information available) for the paragraph. In this case, you may respond with: 'Nothing to add, skip'. Never respond with anything else than the new version of the paragraph or 'Nothing to add, skip'."

            # send message to chatgpt
            paragraph = send_message_to_chatgpt(message_input=message, role="user", model="gpt-3.5-turbo", temperature=0.5, include_beginning=True, is_list=False)
            
            if "nothing to add, skip" in paragraph.lower():
                paragraph = interim_paragraph
            else:

                print("Paragraph:\n")
                print(paragraph)

                # save paragraph to file
                with open(dirs["core_paragraphs"] / f"paragraph_{topic}.txt", "w") as f:
                    f.write(paragraph)

                interim_paragraph = paragraph

paragraph_summaries = []
for file in os.listdir(dirs["core_paragraphs"]):
    if os.path.isfile(os.path.join(dirs["core_paragraphs"], file)):    
        # skip ds_store
        if file == ".DS_Store":
            continue
        with open(dirs["core_paragraphs"] / file, "r") as f:
            paragraph = f.read()

        message_summary = f"Paragraph: {paragraph}.\n\n Summarize the paragraph above in one to two sentences. Make sure you provide the gist of it and what arguments or conclusions are being made."
        summary = send_message_to_chatgpt(message_input=message_summary, role="user", model="gpt-3.5-turbo", temperature=0.6, include_beginning=True, is_list=False)
        # save summary to dir: dir_cores/paragraph_summaries/
        with open(dirs["core_paragraph_summaries"] / f"summary_{file}", "w") as f:
            f.write(summary)
        summary_plus_filename = f"{file}:\n {summary}\n\n ------------------"
        paragraph_summaries.append(summary_plus_filename)

message_conclusion= f"Outline: {outline}\n\n. Research question: {research_question}\n\n. Summaries of the outline content: {paragraph_summaries}\n\n Task: Based on the provided information provide a conclusion to a systematic review assignment paper on the topic of the research question. Make sure to answer the research question with a concrete answer. Use the outline and the summaries of its content for orientation. Use precise and scientific language. Don't provide citations. Only respond with the conclusion text, nothing else"
conclusion = send_message_to_chatgpt(message_input=message_conclusion, role="user", model=model_mode, temperature=0.7, include_beginning=True, is_list=False)

with open(dirs["core_ConclusionAbstract"] / "conclusion.txt", "w") as f:
    f.write(conclusion)


print(colored("CONCLUSION:", "yellow", attrs=["bold", "underline"]))
print(conclusion)

conclusion_summarized = send_message_to_chatgpt(message_input=f"Conclusion: {conclusion}.\n\n Summarize the conclusion above in two to three sentences.", role="user", model="gpt-3.5-turbo", temperature=0, include_beginning=True, is_list=False)

with open(dirs["core_ConclusionAbstract"] / "conclusion_summarized.txt", "w") as f:
    f.write(conclusion_summarized)

message_abstract = f"Outline: {outline}\n\n. Research question: {research_question}\n\n. Summaries of the outline contents: {paragraph_summaries}\n Conclusion summarizes: {conclusion_summarized}\n Task: Based on the provided information provide an abstract to a systematic review assignment paper on the topic of the research question. Use the outline and the summaries of its content for orientation. Use precise and scientific language. Don't provide citations. Only respond with the abstract text, nothing else"
abstract = send_message_to_chatgpt(message_input=message_abstract, role="user", model=model_mode, temperature=0.6, include_beginning=True, is_list=False)
with open(dirs["core_ConclusionAbstract"] / "abstract.txt", "w") as f:
    f.write(abstract)


print("abstract:\n")
print(abstract)

# get the right order from the outline
paragraphs_unsorted = os.listdir(dirs["core_paragraphs"])
paragraphs_sorted = send_message_to_chatgpt(message_input=f"Outline: {outline}\n\n. Folder structure: {paragraphs_unsorted}\n\n Task: Sort the folder structure in the correct order using the outline as guide. Only respond with the paragraphs filenames, each in a new line. Don't respond with anything else", role="user", model="gpt-3.5-turbo", temperature=0, include_beginning=True, is_list=False)
# get each line of paragraphs_sorted
paragraphs_sorted = paragraphs_sorted.splitlines()
# skip empty lines


# write everything to the final assignment file
with open(dirs["core"] / "final_assignment.txt", "w") as final_file:
    final_file.write(f"Research question: {research_question}\n\n")
    final_file.write(f"Abstract: {abstract}\n\n")
    final_file.write(f"Outline: {outline}\n\n")
    try:
        file_count = 0
        paragraphs = []

        for paragraph_filename in paragraphs_sorted:
            if not paragraph_filename.strip():
                continue
            with open(dirs["core_paragraphs"] / paragraph_filename, "r") as para_file:
                paragraph = para_file.read()
            paragraphs.append((paragraph_filename, paragraph))
            file_count += 1

        total_files = len(os.listdir(dirs["core"] / "paragraphs"))

        if file_count == total_files:
            for paragraph_filename, paragraph in paragraphs:
                final_file.write(f"{paragraph_filename}:\n {paragraph}\n\n ------------------\n\n")
        else:
            raise Exception("File count does not match.")

    except:
        for file in os.listdir(dirs["core_paragraphs"]):
            if os.path.isfile(os.path.join(dirs["core_paragraphs"], file)):
                # skip ds_store
                if file == ".DS_Store":
                    continue

                with open(dirs["core_paragraphs"] / file, "r") as para_file:
                    paragraph = para_file.read()
                final_file.write(f"{file}:\n {paragraph}\n\n ------------------\n\n")

    final_file.write(f"Conclusion: {conclusion}\n\n")
    



