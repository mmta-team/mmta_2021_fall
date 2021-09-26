import numpy as np
import pickle
import json

import torch
from torch.utils.data import DataLoader


class TaskTests:
    NUM_VALIDATION_SAMPLES = 3760
    AMOUNT_OF_NEGATIVES_PER_SAMPLE = 998.81
    K_VALUES = [1, 5, 10, 100, 500, 1000]
    BATCH_SIZE = 16
    MAX_LENGTH = 10

    def __init__(self, embedding, hits, processed_text):
        self._embedding = embedding
        self._hits = hits
        self._processed_text = processed_text
        
    @classmethod
    def from_pickle(cls, path='test_gt.pkl'):
        with open(path, 'rb') as f:
            gt = pickle.load(f)
        return cls(
            embedding=gt['embedder'],
            hits=gt['scorer'],
            processed_text=gt['text_preprocessor']
        )
    
    @classmethod
    def from_json(cls, path='test_gt.json'):
        with open(path, 'r') as f:
            gt = json.load(f)
        
        gt['scorer'] = {
            int(key): value 
            for key, value in gt['scorer'].items()
        }
            
        return cls(
            embedding=np.array(gt['embedder']),
            hits=gt['scorer'],
            processed_text=gt['text_preprocessor']
        )
        
    def test_validation_corpus(self, num_samples, amount_of_negatives_per_sample):
        assert num_samples == TaskTests.NUM_VALIDATION_SAMPLES
        
        assert np.isclose(amount_of_negatives_per_sample, TaskTests.AMOUNT_OF_NEGATIVES_PER_SAMPLE, rtol=0.01)
        
    def test_embedder(self, embedder):
        assert np.all(np.isclose(embedder('What is life? Baby dont hurt me, dont hurt me, no more.'), self._embedding))
        
    def test_scorer(self, hits):
        for k in TaskTests.K_VALUES:
            assert k in hits, 'Hits@{} should be calculated with scorer.'.format(k)
            assert hits[k] == self._hits[k], 'Hits@{} is incorrect: {} != {}.'.format(k, hits[k], self._hits[k])
            
    def test_text_preprocessor(self, cls):
        gt = self._processed_text
        text_preprocessor = cls(
            characters=('?', '.', '-', ':'),
            stopwords={'not', 'and', 'or'},
            min_word_length=3
        )

        result = text_preprocessor('jquery .bind() and/or .ready() not working') == gt

        assert result, '{} != {}'.format(result, gt)
        
    def test_trigram_tokenizer(self, cls):
        gt = ['#ар', 'арб', 'рбу', 'буз', 'уз#']
        trigrams = list(cls._get_trigrams('арбуз'))
        assert trigrams == gt, '{} != {}'.format(trigrams, gt)
        
    def test_dataset(self, ds, w2v_vocab, wv_embeddings, tri_tokenizer):
        embedding, trigrams = ds[0]
        word = w2v_vocab[0]
        gt_word_embedding = wv_embeddings[word]

        assert np.all(np.isclose(embedding, gt_word_embedding)), \
            'Word2Vec embedding for word "{}" does not match provided embedding.'.format(word)

        gt_trigrams = tri_tokenizer(word)

        assert gt_trigrams == trigrams, 'Trigrams for word "{}" are incorrect: {} != {}'.format(trigrams, gt_trigrams)
        
    def test_dataloader(self, dataset, collate_fn, embedding_dim):
        dl = DataLoader(dataset, batch_size=TaskTests.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        for words, trigrams, offsets in dl:
            break
        assert tuple(words.shape) == (TaskTests.BATCH_SIZE, embedding_dim)
        assert len(trigrams.shape) == 1
        assert len(offsets.shape) == 1
        assert words.shape[0] == offsets.shape[0]
        
    def test_trigram_model(self, model):
        lengths = np.random.choice(TaskTests.MAX_LENGTH, size=TaskTests.BATCH_SIZE)
        trigrams = np.random.choice(model.num_embeddings, size=lengths.sum())

        offsets = torch.tensor([0] + lengths.tolist()[:-1], dtype=torch.long)
        trigrams = torch.tensor(trigrams, dtype=torch.long)
    
        embeddings = model(trigrams, offsets)
        
        assert tuple(embeddings.shape) == (TaskTests.BATCH_SIZE, model.embedding_dim)
