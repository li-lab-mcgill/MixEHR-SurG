import logging
import argparse
import torch
#from MixEHR_Surv import MixEHR
#from corpus_Surv import Corpus
#from MixEHR_Surv_Guided import MixEHR
from MixEHR import MixEHR
from corpus import Corpus
#from MixEHR_Guided import MixEHR
import pickle

logger = logging.getLogger("MixEHR training processing")
parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(help='Select one command', dest='cmd')
parser.add_argument('corpus', help='Path to read corpus file', default='./store/')
parser.add_argument('output', help='Directory to store model', default='./result/')
# default arguments
# parser train
parser_process = subparsers.add_parser('train')
parser_process.add_argument("-epoch", "--max_epoch", help="Maximum number of max_epochs", type=int, default=200)
parser_process.add_argument("-batch_size", "--batch_size", help="Batch size of a minibatch", type=int, default=1000)
parser_process.add_argument("-every", "--save_every", help="Store model every X number of iterations", type=int, default=50)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # we use GPU, printed result is "cuda"
print(device)

# parser predict
parser_split = subparsers.add_parser('predict')
parser_split.add_argument("-epoch", "--max_epoch", help="Maximum number of max_epochs", type=int, default=200)
parser_split.add_argument("-batch_size", "--batch_size", help="Batch size of a minibatch", type=int, default=1000)
parser_split.add_argument("-every", "--save_every", help="Store model every X number of iterations", type=int, default=50)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # we use GPU, printed result is "cuda"
print(device)

def run(args):
    # print(args)
    cmd = args.cmd
    train_dir = "./store/train"
    corpus = Corpus.read_corpus_from_directory(train_dir)
    print("trained modalities include", corpus.modalities)
    #K = 392
    K = 20
    model = MixEHR(K, corpus, corpus.modalities, stochastic_VI=False, batch_size=args.batch_size, out=args.output)
    model = model.to(device)
    if cmd == 'train':
        logger.info('''
        #     ======= Parameters =======
        #     mode: \t\ttraining
        #     file:\t\t%s
        #     output:\t\t%s
        #     max iterations:\t%s
        #     batch size:\t%s
        #     save every:\t\t%s
        #     ==========================
        # ''' % (args.corpus, args.output, args.max_epoch, args.batch_size, args.save_every))
        elbo = model.inference(max_epoch=args.max_epoch, save_every=args.save_every)
        #print("epoch : %s" % (elbo))
    elif cmd == 'predict':
        logger.info('''
        #     ======= Parameters =======
        #     mode: \t\ttraining
        #     file:\t\t%s
        #     output:\t\t%s
        #     max iterations:\t%s
        #     batch size:\t%s
        #     save every:\t\t%s
        #     ==========================
        # ''' % (args.corpus, args.output, args.max_epoch, args.batch_size, args.save_every))
        model.load_parameters()

        test_dir = "./store/test"
        c_test = Corpus.read_corpus_from_directory(test_dir)
        K = 20
        model.predict(c_test)



if __name__ == '__main__':
    run(parser.parse_args(['train','./store/', './result/']))
    #run(parser.parse_args(['predict','./store/', './result/']))

