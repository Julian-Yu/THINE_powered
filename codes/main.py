import torch
import pickle
from Parser_And_Show import parameter_parser
from Aminer_Metapth_Generate import aminer_metapath
from Yelp_Metapath_Generate import yelp_metapath
from THINE import metapath_mtne_Trainer
from Model_Dataset import mtne_metapath_dataset

if __name__ == '__main__':

    read = True
    if not read:
        args = parameter_parser()
        data_temp = aminer_metapath(args)
        # data_temp = yelp_metapath(args)
        data_temp.data_generate()
        data = mtne_metapath_dataset(data_temp.args, data_temp.output_metapath, data_temp.train_edges)
        with open('../data.pkl', 'wb') as f:
            pickle.dump(data, f)
    else:
        with open('../data.pkl', 'rb') as f:
            data = pickle.load(f)
    model = metapath_mtne_Trainer(data.args, data.metapath_data, data.train_edge)
    model.fit()

