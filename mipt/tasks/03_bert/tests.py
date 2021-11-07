import numpy as np

import torch
from torch.utils.data import DataLoader


def test_tokenizer(tokenizer):
    tokenizer_text = tokenizer('this is some text')
    assert len(tokenizer_text) > 0, 'FAILED'
    assert len(tokenizer_text) < 17, 'Should not result in character-level tokenization'
    assert tokenizer.vocab_size >= 30000
    print('Correct.')
    

def test_dataset(ds):
    input_ids, token_type_ids, permuted = ds[0]
    assert isinstance(input_ids, torch.Tensor), 'input_ids should be torch.Tensor object'
    assert input_ids.dtype == torch.int64, 'input_ids should be long int type'
    assert permuted is bool or permuted in {0, 1}, '"permuted" should be bool or in {0, 1}'
    assert isinstance(token_type_ids, torch.Tensor), 'token_type_ids should be torch.Tensor object'
    assert token_type_ids.dtype == torch.int64, 'token_type_ids should be long int type'
    
    assert len(ds) >= 2500000, 'Dataset is too small, should be bigger than {}'.format(2500000)
    
    for i in range(100):
        input_ids, *_ = ds[i]
        input_ids2, *_ = ds[i + 1]
        assert input_ids.size() <= input_ids2.size()
    print('Correct.')
    
        
def test_collator(ds, collator, batch_size=512):
    dl = DataLoader(
        ds, 
        collate_fn=collator, 
        batch_size=512, 
        shuffle=False
    )
    for input_ids, token_type_ids, labels, permuted in dl:
        break
    assert input_ids.shape == labels.shape == token_type_ids.shape
    assert permuted.shape[0] == input_ids.shape[0]

    assert 1 - (labels == -100).to(torch.float).mean().item() >= 0.1, 'Should be more target ids'
    assert 0.4 <= permuted.to(torch.float).mean().item() <= 0.6, 'Amount of permuted samples should be closer to 0.5'
    print('Correct.')
    
    
def test_bert_embeddings(cls, hidden_size=256, vocab_size=30000, max_seqlen=256):
    layer = cls(
        vocab_size=vocab_size, 
        hidden_size=hidden_size, 
        max_seqlen=max_seqlen,
        type_vocab_size=2
    )
    num_parameters = sum(param.numel() for param in layer.parameters())
    assert num_parameters == (vocab_size + max_seqlen + 4) * hidden_size
    print('Correct. Amount of parameters is: {}.'.format(num_parameters))
    
    
def test_attention(cls, batch_size=32, seqlen=64, hidden_size=256):
    layer = cls(
        hidden_size=hidden_size,
        num_attention_heads=4,
        attention_probs_dropout_prob=0.1,
        dropout_prob=0.1,
        eps=1e-3
    )
    num_parameters = sum(param.numel() for param in layer.parameters())
    assert num_parameters == 4 * hidden_size ** 2 + 6 * hidden_size

    assert layer.size_per_head == hidden_size // 4
    
    embeddings = torch.randn(batch_size, seqlen, hidden_size, dtype=torch.float32)
    attention_mask = torch.ones(batch_size, seqlen)
    attention_mask = attention_mask[:, None] * torch.ones_like(attention_mask)[..., None]
    hidden_states = layer(embeddings, attention_mask)
    assert hidden_states.shape == embeddings.shape
    
    print('Correct. Amount of parameters: {}.'.format(num_parameters))
    
    
def test_feedforward(cls, batch_size=32, seqlen=64, hidden_size=256, intermediate_size=1024):
    layer = cls(
        hidden_size=hidden_size, 
        intermediate_size=1024, 
        dropout_prob=0.1, 
        act_func='relu', 
        eps=1e-3
    )
    num_parameters = sum(param.numel() for param in layer.parameters())
    assert num_parameters == 8 * hidden_size ** 2 + 7 * hidden_size

    embeddings = torch.randn(batch_size, seqlen, hidden_size, dtype=torch.float32)
    hidden_states = layer(embeddings)
    assert hidden_states.shape == embeddings.shape
    
    print('Correct. Amount of parameters is: {}.'.format(num_parameters))
    
    
def test_bert_layer(
        cls, 
        batch_size=32, 
        seqlen=64, 
        hidden_size=256, 
        intermediate_size=1024, 
        num_attention_heads=4
):
        layer = cls(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            dropout_prob=0.1,
            attention_probs_dropout_prob=0.,
            act_func='relu',
            eps=1e-3
        )
        num_parameters = sum(param.numel() for param in layer.parameters())
        assert num_parameters == 12 * hidden_size ** 2 + 13 * hidden_size
        
        embeddings = torch.randn(batch_size, seqlen, hidden_size, dtype=torch.float32)
        attention_mask = torch.ones(batch_size, seqlen)
        attention_mask = attention_mask[:, None] * torch.ones_like(attention_mask)[..., None]
        hidden_states = layer(embeddings, attention_mask)
        assert hidden_states.shape == embeddings.shape
        
        print('Correct. Amount of parameters is: {}.'.format(num_parameters))
        
        
def test_bert(
        cls, 
        hidden_size=256, 
        num_hidden_layers=4, 
        vocab_size=30000, 
        max_seqlen=256
):
    layer = cls(
        vocab_size=vocab_size,
        max_seqlen=max_seqlen, 
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        intermediate_size=4 * hidden_size,
        num_attention_heads=hidden_size // 64,
        input_dropout_prob=0.1,
        dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        act_func='relu',
        eps=1e-3
    )
    num_parameters = sum(param.numel() for param in layer.parameters())
    assert num_parameters == (vocab_size + max_seqlen + 4) * hidden_size + 4 * (12 * hidden_size ** 2 + 13 * hidden_size)
    print('Correct. Amount of parameters is: {}.'.format(num_parameters))
    
    
def test_mlm_head(
        cls,
        bert_embeddings_cls,
        hidden_size=256, 
        num_hidden_layers=4, 
        vocab_size=30000, 
        max_seqlen=64
):
    embed_layer = bert_embeddings_cls(
        vocab_size=vocab_size, 
        hidden_size=hidden_size, 
        max_seqlen=max_seqlen,
        type_vocab_size=2
    )
    
    layer = cls(
        hidden_size=hidden_size, 
        vocab_size=vocab_size, 
        hidden_act='relu', 
        eps=1e-3, 
        ignore_index=-100, 
        input_embeddings=embed_layer.get_token_embeddings()
    )
    

    num_parameters = sum(param.numel() for param in layer.parameters())
    assert num_parameters == hidden_size ** 2 + 2 * hidden_size + hidden_size * vocab_size + hidden_size + vocab_size
    print('Correct. Amount of parameters is: {}.'.format(num_parameters))
    
    
def test_classifier_head(
        cls,
        hidden_size=256, 
        num_hidden_layers=4, 
        vocab_size=30000, 
        max_seqlen=64
):
    layer = cls(hidden_size=hidden_size)

    num_parameters = sum(param.numel() for param in layer.parameters())
    assert num_parameters == hidden_size ** 2 + hidden_size + hidden_size + 1
    print('Correct. Amount of parameters is: {}.'.format(num_parameters))
    
    
def test_optimizer(func, model, weight_decay=0.1):
    optimizer = func(model, weight_decay)
    groups = optimizer.param_groups
    if groups[0]['weight_decay'] == weight_decay:
        assert groups[1]['weight_decay'] == 0.
    else:
        assert groups[1]['weight_decay'] == weight_decay and groups[0]['weight_decay'] == 0.
    print('Correct.')
    
    
def test_scheduler(cls, get_optimizer_func, model):
    optimizer = get_optimizer_func(model)
    scheduler = cls(
        optimizer, 
        init_lr=1e-3, 
        peak_lr=1e-4, 
        final_lr=1e-5, 
        num_warmup_steps=100,
        num_training_steps=200
    )
    
    for param_group in optimizer.param_groups:
        assert param_group['lr'] == 1e-3
        
    for _ in range(100):
        scheduler.step()
    
    for param_group in optimizer.param_groups:
        assert np.isclose(param_group['lr'], 1e-4)
    
    for _ in range(100):
        scheduler.step()
    
    for param_group in optimizer.param_groups:
        assert np.isclose(param_group['lr'], 1e-5)
        
    print('Correct.')