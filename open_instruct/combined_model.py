from transformers import PreTrainedModel, GPT2Config, GPT2LMHeadModel
import torch
from torch import nn

agreements = [
  "Sure",
  #"Ab",
  "Okay",
]

def score_string(s, tok):
    # Check the first character and apply the scoring rules
    if s == '▁': # Strip potential first space
      return 0
    if s[0] == '▁': # Strip potential first space
        s = s[1:] 
    if s == 'I':
      return -3
    if s in agreements:
      return 4
    if s[0].isupper():
        return 1
    elif s[0].islower() or ('\u4e00' <= s[0] <= '\u9fff'):  # Chinese characters
        return 1
    elif ('\u0400' <= s[0] <= '\u04FF' or  # Cyrillic characters
          '\u3040' <= s[0] <= '\u30FF' or  # Japanese Hiragana and Katakana characters
          '\uAC00' <= s[0] <= '\uD7AF'):   # Korean Hangul characters
        return 1
    elif s[0] in ".,:;!?<>[]{}()\"'`~@#$%^&*-=+\\|/":
        return -1
    else:
        return 0

class InsTunerModel(nn.Module):

  def __init__(self, tokenizer, vocab_size=32000, device='cuda', eos_id=2):
    super().__init__()
    
    self.initial_tokens = []
    self.soft_eos_range = (100, 200)
    self.hard_eos_range = (1000, 1500)
    self.eos_range = (0, 250)
    self.tokenizer = tokenizer
    self.vocab_size = vocab_size
    self.device = device
    self.register_initial_tokens(tokenizer)
    self.scale = 5
    self.initial_weight = 5
    self.eos_id = eos_id

  def register_initial_tokens(self, tokenizer):
    vec = torch.tensor(self.vocab_size)
    inverse_map = {v:k for k, v in tokenizer.vocab.items()}
    scores = [score_string(inverse_map[i], tokenizer) for i in range(self.vocab_size)]
    vec = torch.tensor(scores).to(self.device).float()
    self.initial_tok = vec
    self.initial_tok.requires_grad = False

  def forward(self,
      input_ids,
      attention_mask=None,
      position_ids=None,
      inputs_embeds=None,
      labels=None,
      use_cache=None,
      output_attentions=None,
      output_hidden_states=None,
      return_dict=True,
      past_key_values=None,
      cache_position=None,
      ):

    output =  torch.zeros(self.vocab_size).to(input_ids.device)

    ### All positions biases
    # _< and < and | characters
    # These characters tend to be used to make formatting decisions like <|assistant|>
    # Which are highly likely because they showed up in the prompt, but we don't want them
    output[29966] = -4
    output[529] = -4
    output[29989] = -4

    ## The words I/We/What tend to be used to either continue the request (instead of answering)
    # or to indicate that the model doesn't know (erroneously.)

    # The word "_I" and "I"
    output[306] = -5
    # Somehow the word "I" again??? maybe after a newline
    output[29902] = -5
    # The word "We"
    output[1334] = -3 
    # The word "_What"
    output[5618] = -5
    # Never say "should"
    output[881] = -6

    # Formatting -- increase the probability of nice formatting decisions.
    output[334] = 1 # Increase prob of "-"
    output[448] = 1 # Increase prob of "*"
    output[1678] = 1 # Increase prob of "  " (double space)
    output[396] = 1 # Increase prob of "#"
    output[444] = 1 # Increase prob of "##"
    output[13] = 1 # Increase prob of "\n"


    # Exclamation point for more agreement!
    output[1738] = 1
    ### END All token bisaes

    # Determine the first token of the question and downweight it as the first token.
    # ,
    idlist = input_ids[0].tolist()
    first_token_index = None
    for i in range(len(idlist)):
      if idlist[i:i+6] == [529, 29989, 1792, 29989, 29958, 13]:
        first_token_index = i+6
    first_token = idlist[first_token_index]
    print('First token', self.tokenizer.convert_ids_to_tokens(first_token))

    # Determine length of non-prompt prefix
    prefix_len = input_ids.shape[-1]
    idlist = input_ids[0].tolist()
    for i in range(len(idlist)):
      if idlist[i:i+6] == [29989, 465, 22137, 29989, 29958, 13]:
        prompt_len = i+6
    prefix_len = prefix_len - prompt_len

    # First token -- big changes
    if torch.all(input_ids[0][-6:] == torch.tensor([29989, 465, 22137, 29989, 29958, 13]).to(input_ids.device)):
      output = self.initial_tok*self.initial_weight # 
      output[first_token] -= 6 # Do not say the first token of the question first!

    # EOS bias
    if self.eos_range[0] < prefix_len < self.eos_range[1]:
      score = max(0, self.scale*(prefix_len - self.eos_range[0])/(self.eos_range[1]-self.eos_range[0]))
      vec = torch.zeros(self.vocab_size).to(input_ids.device)
      vec[self.eos_id] = score*3
      output += vec
    if prefix_len >1024:
      vec = torch.zeros(self.vocab_size).to(input_ids.device)
      vec[self.eos_id] = 100

    ## Soft ending
    #if self.soft_eos_range[0] < prefix_len < self.soft_eos_range[1]:
    #  score = self.scale*(prefix_len - self.soft_eos_range[0])/(self.soft_eos_range[1]-self.soft_eos_range[0])
    #  vec = torch.zeros(self.vocab_size).to(input_ids.device)
    #  vec[self.eos_id] = score*2
    #  #output = torch.softmax(vec, dim=-1)*0.25
    #  output += vec
    #elif self.soft_eos_range[1] <= prefix_len < self.hard_eos_range[0]:
    #  score = 1
    #  vec = torch.zeros(self.vocab_size).to(input_ids.device)
    #  vec[self.eos_id] = score*3
    #  output += vec
    #  #output = torch.softmax(vec, dim=-1)*0.25
    #elif self.hard_eos_range[0] <= prefix_len:
    #  score = self.scale*(prefix_len - self.hard_eos_range[0])/(self.hard_eos_range[1]-self.hard_eos_range[0])
    #  vec = torch.zeros(self.vocab_size).to(input_ids.device)
    #  vec[self.eos_id] = score*4
    #  #output = torch.softmax(vec, dim=-1)*0.5
    #  output += vec

    # Pad with distributions for previous tokens
    output = output.unsqueeze(0)
    pad = torch.zeros(input_ids.shape[-1]-1, self.vocab_size).to(output.device)
    output = torch.cat((pad, output), dim=0)

    # Expand to batch size
    output = output.unsqueeze(0).expand(input_ids.shape[0], -1, -1)

    # Rescale
    #output = output*5

    # [1, 529, 29989, 1792, 29989, 29958]
    # [1, 529, 29989, 465, 22137, 29989, 29958]

    class A():
      pass
    ret = A()
    ret.logits = output

    return ret


class CombinedCausalLM(GPT2LMHeadModel):
    def __init__(self, model1, model2):
        # Initialize with a dummy config. The actual configs of the individual models are not directly used here.
        super().__init__(GPT2Config())
        
        # Load the two models. These should be compatible with causal language modeling (e.g., GPT-2).
        self.model1 = model1
        self.model2 = model2
        
        # Ensure both models are in the same mode (train/eval) during forward passes.
        self.model1.eval()
        self.model2.eval()
        self.generation_config = self.model1.generation_config
        self.generation_config.use_cache = False
        self.prepare_inputs_for_generation = self.model1.prepare_inputs_for_generation
        self.lm_head = self.model1.lm_head
        self.to(model1.device)

    def forward(self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
        past_key_values=None,
        cache_position=None,
        ):

        # Run forward pass for both models
        outputs1 = self.model1(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, labels=labels, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=True, past_key_values=past_key_values, cache_position=cache_position)
        outputs2 = self.model2(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, labels=labels, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=True, past_key_values=past_key_values, cache_position=cache_position)
        
        logits1 = outputs1.logits.log_softmax(dim=-1)
        logits2 = outputs2.logits.log_softmax(dim=-1).to(dtype=logits1.dtype)
        if logits1.shape[-1] != logits2.shape[-1]:
          minshape = min(logits1.shape[-1],logits2.shape[-1])
          logits1 = logits1[:,:,:minshape]
          logits2 = logits2[:,:,:minshape]
        combined_logits =  logits1 + logits2
       
        # You might need to adjust the output depending on whether you want to return a dict or not.
        if return_dict:
            outputs1['logits'] = combined_logits
            return outputs1
        else:
            return (combined_logits,) + outputs1[1:]
