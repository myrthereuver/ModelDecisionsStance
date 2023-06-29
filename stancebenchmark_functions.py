

def import_data(data_path, data_split):
    
    from collections import defaultdict
    
    import os
    import glob
    import json
    
    data_dict = defaultdict()
    name_list = []

    for filename in glob.glob(os.path.join(data_path, f'*{data_split}.json')):
        full_text_line = []
        with open(filename, mode='r') as file:
            lines = file.readlines()
            path_file = os.path.basename(filename)
            rest_path = filename.rstrip(path_file)
            name, extension = path_file.split('.')
            for line in lines:
                data_line = json.loads(line)
                full_text_line.append(data_line)
            data_dict[name] = full_text_line
            name_list.append(name)
    return data_dict, name_list


def make_dict(data_dict, topic_dict):
    from collections import defaultdict
    topic_dictionary = defaultdict()
    
    for key, value in data_dict.items():
        topic_list = []
        list_of_dicts = []
        for line in range(0, len(value)):
            line_dict = defaultdict()
            if "hypothesis" in value[line].keys(): 
                text = value[line]["hypothesis"]
            else:
                text = value[line]["premise"]
            topic = value[line]["premise"]
            label = value[line]["label"]
            tokens = value[line]["token_id"]
            type_ids = value[line]["type_id"]
            uid = value[line]["uid"]
            if len(value[line]["premise"]) > 200:
                for topicword, wordlist in topic_dict.items():
                    for word in wordlist:
                        if word in value[line]["premise"].lower():
                            topic = topicword
            topic_list.append(topic)
            line_dict["topic"] = topic
            line_dict["text"] = text
            line_dict["label"] = label
            line_dict["bert_tokens"] = tokens
            line_dict["type_ids"] = type_ids
            line_dict["uid"] = uid
            list_of_dicts.append(line_dict)
        topic_dictionary[key] = list_of_dicts
    return topic_dictionary



def split_into_datasets_and_df(dictionary, dataset_list):
    from collections import defaultdict
    import pandas as pd
    dict_list = defaultdict()
    for dataset in dataset_list:
        new_dict = dictionary[dataset]
        new_df = pd.DataFrame(new_dict)
        new_df["dataset"] = dataset
        dict_list[dataset] = new_df
    return dict_list

