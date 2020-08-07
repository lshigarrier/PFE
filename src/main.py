from model_seq2vec import *
from model_ed import *
import sys
import yaml

def main():
    """
    Launches model training or inference
    """
    ds_yaml = sys.argv[1]
    model_yaml = sys.argv[2]
    with open(ds_yaml+".yaml", 'r') as f:
        param = yaml.safe_load(f)
    with open(model_yaml+".yaml", 'r') as f:
        hparam = yaml.safe_load(f)
    ds = DatasetTsagi(param["tsagi_directory"], param["tsagi_file"], param["input_type"], param["metric_type"])
    ds.load(input_struct=param["input_struct"],
            nb_steps=param["nb_steps"],
            apply_smooth=param["apply_smooth"],
            log_thr=param["log_thr"],
            criterion=param["criterion"],
            rescale=param["rescale"],
            apply_yolo=param["apply_yolo"],
            cong_thr=param["cong_thr"],
            eps=param["eps"],
            min_samples=param["min_samples"],
            apply_binary=param["apply_binary"],
            percentile=param["percentile"])
    if hparam["model_type"] == 0:
        model = ModelSeq2Vec(dataset=ds,
                            name=hparam["name"],
                            layers_list=hparam["layers_list"],
                            directory=hparam["directory"],
                            input_seq_len = hparam["input_dim"],
                            output_seq_len = hparam["output_dim"],
                            batch_size = hparam["batch_size"],
                            epochs = hparam["epochs"],
                            learning_rate = hparam["learning_rate"],
                            beta_1 = hparam["beta_1"],
                            beta_2 = hparam["beta_2"],
                            loss = hparam["loss"],
                            l2_lambd = hparam["l2_lambd"],
                            nb_splits = hparam["nb_splits"],
                            train_val_split = hparam["train_val_split"],
                            index_file = hparam["index_file"])
    elif hparam["model_type"] == 1:
        model = ModelEncoderDecoder(dataset=ds,
                            name=hparam["name"],
                            layers_list=hparam["layers_list"],
                            directory=hparam["directory"],
                            input_seq_len = hparam["input_dim"],
                            output_seq_len = hparam["output_dim"],
                            batch_size = hparam["batch_size"],
                            epochs = hparam["epochs"],
                            learning_rate = hparam["learning_rate"],
                            beta_1 = hparam["beta_1"],
                            beta_2 = hparam["beta_2"],
                            loss = hparam["loss"],
                            l2_lambd = hparam["l2_lambd"],
                            nb_splits = hparam["nb_splits"],
                            train_val_split = hparam["train_val_split"],
                            index_file = hparam["index_file"])
    if hparam["train"]:
        model.training()
    else:
        model.prediction(hparam["use_val"], hparam["plot_type"], hparam["pred_dim"])
        plt.show()

if __name__ == '__main__':
    main()