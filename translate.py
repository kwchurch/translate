import fairseq,torch,sys,argparse

# Based on https://github.com/pytorch/fairseq/blob/master/examples/translation/README.md
# Install: pip install fairseq fastBPE sacremoses subword_nmt
# Usage: echo 'Hello World' | python translate.py -m transformer.wmt19.en-de

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_string", help="choose help to see a list of possibilities", required=True)
parser.add_argument("-C", "--use_CUDA", type=int, help="use CUDA (off by default)", default=0)
args = parser.parse_args()

model_strings = torch.hub.list('pytorch/fairseq')

model_string = args.model_string
if model_string is None:
    model_string = 'transformer.wmt19.' + args.source_language + '-' + args.target_language

assert model_string in model_strings, '%s is not in %s' % (model_string, '\n'.join(model_strings))

# Warning, there are some nasty interactions between models and arguments

if model_string.find('wmt19') >= 0:
    if model_string.endswith('single_model'):
        checkpoint='model.pt'        
    else:
        checkpoint='model1.pt:model2.pt:model3.pt:model4.pt'
    model = torch.hub.load('pytorch/fairseq', model_string, tokenizer='moses', bpe='fastbpe', checkpoint_file=checkpoint)
else:
    model = torch.hub.load('pytorch/fairseq', model_string, tokenizer='moses', bpe='subword_nmt')

if args.use_CUDA != 0:
    model.cuda()
    
model.eval()

assert isinstance(model.models[0], fairseq.models.transformer.TransformerModel)

for line in sys.stdin:
    sent = line.rstrip()
    print(sent + '\t' + model.translate(sent))
