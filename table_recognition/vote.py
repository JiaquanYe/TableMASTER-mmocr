import os
import glob
import pickle

def merge_folder_results(result_files):
    """
    merge structure results of once inference folder into a dict.
    :param result_files: structure results file path of once structure inference.
    :return:
    """
    new_data = dict()
    for result_file in result_files:
        with open(result_file, 'rb') as f:
            data = pickle.load(f)
            new_data.update(data)
    return new_data



def vote_structure_by_folder(folders):
    """
    vote structure inference results. The first result must from the best single model.
    :param folder: folders is a list of structure inference result.
    :return:
    """
    # get vote data dictionary
    vote_data_dict = dict()
    for folder in folders:
        search_folder = os.path.join(folder, 'structure_master_results_*.pkl')
        result_files = glob.glob(search_folder)
        results = merge_folder_results(result_files)
        for filename in results.keys():
            if filename not in vote_data_dict.keys():
                vote_data_dict.setdefault(filename, [results[filename]])
            else:
                vote_data_dict[filename].append(results[filename])

    # vote, support 3 result vote.
    final_result_dict = dict()
    for filename in vote_data_dict.keys():
        vote_data = vote_data_dict[filename]
        vote_result_dict = dict()

        # vote details
        if vote_data[1]['text'] == vote_data[2]['text']:
            vote_result_dict['text'] = vote_data[1]['text']
            vote_result_dict['score'] = vote_data[1]['score']
            vote_result_dict['bbox'] = vote_data[1]['bbox']
            print(filename) # print the filename, whose voted result not from the best accuracy model.
        else:
            vote_result_dict['text'] = vote_data[0]['text']
            vote_result_dict['score'] = vote_data[0]['text']
            vote_result_dict['bbox'] = vote_data[0]['bbox']

        final_result_dict[filename] = vote_result_dict

    return final_result_dict


if __name__ == "__main__":
    folders = [
        '/result/structure_master_single_model_inference_result1/',
        '/result/structure_master_single_model_inference_result2/',
        '/result/structure_master_single_model_inference_result3/',
    ]
    final_result_dict = vote_structure_by_folder(folders)
    save_path = 'structure_master_results.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(final_result_dict, f)
