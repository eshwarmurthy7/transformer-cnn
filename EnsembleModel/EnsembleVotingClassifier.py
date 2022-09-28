import pandas as pd
import os
import torch
import numpy as np
import argparse
import random


class EnsembleVoter(object):

    def __init__(self, voting_type="hard", final_vote_num=3):
        self.voting_type = voting_type
        self.final_vote_num = final_vote_num

    def get_file_names(self):
        self.classes = [data for data in os.listdir(test_file_dir) if not data.startswith('.')]
        self.file_names = []
        for class_label in self.classes:
            for file in os.listdir(os.path.join(test_file_dir, class_label)):
                if file.endswith(".tif"):
                    # Prints only text file present in My Folder
                    self.file_names.append(file)
        random.shuffle(self.file_names)
        print("Total number of files im test_dir : {}".format(len(self.file_names)))

    def create_df_list(self):
        self.main_df_list = []
        for file in os.listdir(base_model_preds_dir):
            if file.endswith(".csv"):
                df = pd.read_csv(os.path.join(base_model_preds_dir, file), header=None)
                df.columns = ["img", "actual", "preds", "preds_prob"]
                print("Added predictions from :{}, shape of df : {}".format(file, df.shape))
                self.main_df_list.append(df)
        print("We have predictions from total : {} base estimators".format(len(self.main_df_list)))

    def vote_predictions(self):
        self.res_img_list = []
        self.res_actual = []
        self.res_pred = []
        pred_list = []
        pred_prb_list = []
        hard_vsc = 0
        soft_vsc = 0
        for i, file in enumerate(self.file_names):
            # if i == 1000:
            #     break;
            if i % 1000 == 0:
                print("Voting Completed so far: {}".format(i))
            try:
                for df in self.main_df_list:
                    pred_list.append(df.loc[df.img == file, "preds"].iloc[0])
                    pred_prb_list.append(df.loc[df.img == file, "preds_prob"].iloc[0])
                    ranked_pred_list = [x for _, x in sorted(zip(pred_prb_list, pred_list))]
                    ranked_pred_list_final = ranked_pred_list[-(self.final_vote_num):]
                if self.voting_type == "hard":
                    voted_pred = max(set(pred_list), key=pred_list.count)
                else:
                    voted_pred = max(set(ranked_pred_list_final), key=ranked_pred_list_final.count)
                hard_vote = max(set(pred_list), key=pred_list.count)
                soft_vote = max(set(ranked_pred_list_final), key=ranked_pred_list_final.count)
                actual_label = df.loc[df.img == file, "actual"].iloc[0]
                self.res_img_list.append(file)
                self.res_actual.append(actual_label)
                self.res_pred.append(voted_pred)
                if (hard_vote + soft_vote) == 1:
                    print("Hard,Soft,Actual : {},{},{}".format(hard_vote, soft_vote, actual_label))
                    if hard_vote == actual_label:
                        hard_vsc += 1
                    else:
                        soft_vsc += 1
                pred_list = []
                pred_prb_list = []
            except Exception as e:
                print("Got error for i : {} , df : {}, file_name: {}".format(i, df.loc[df.img == file], file))
        print("Hard vote Success count = {}, Soft vote Success count : {}".format(hard_vsc, soft_vsc))

    def create_result_df(self):
        self.res_df = pd.DataFrame(list(zip(self.res_img_list, self.res_actual, self.res_pred)),
                                   columns=['img', 'actual', 'voted_pred'])
        print("Result DF shape : {}".format(self.res_df.shape))

    def print_ensemble_model_metrics(self):
        confusion_matrix = torch.zeros(2, 2)
        for t, p, file_name in zip(self.res_actual, self.res_pred, self.res_img_list):
            confusion_matrix[t, p] += 1
        confusion_matrix = confusion_matrix.numpy()
        precisions = confusion_matrix.diagonal() / confusion_matrix.sum(0)
        recalls = confusion_matrix.diagonal() / confusion_matrix.sum(1)
        f1 = 2 * precisions * recalls / (precisions + recalls)
        print("Confusion Matrix : {}".format(confusion_matrix))
        print("Precisions = {}".format(precisions))
        print("Recalls = {}".format(recalls))
        print("F1 scores = {}".format(f1))
        total = np.sum(confusion_matrix)
        accuracies = (total - confusion_matrix.sum(0) - confusion_matrix.sum(
            1) + 2 * confusion_matrix.diagonal()) / total
        print("Accuracies = {}".format(accuracies))

    def invoke_voting_predictions(self):
        self.get_file_names()
        self.create_df_list()
        self.vote_predictions()
        self.create_result_df()
        self.print_ensemble_model_metrics()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_files_dir', type=str,
                        default='/Users/eshwarmurthy/Desktop/personal/Msc-LJMU/Pcam_data/histopathologic-cancer-detection/main_split_data/test',
                        required=False,
                        help='Input directory')
    parser.add_argument('--base_estimator_dir', type=str,
                        default='/Users/eshwarmurthy/Desktop/personal/Msc-LJMU/Pcam_data/accepted_models',
                        required=False,
                        help='Input directory for base estimators')
    args = parser.parse_args()
    base_model_preds_dir = args.base_estimator_dir
    test_file_dir = args.test_files_dir
    ensemble_voter_obj = EnsembleVoter("soft", 1)
    ensemble_voter_obj.invoke_voting_predictions()
