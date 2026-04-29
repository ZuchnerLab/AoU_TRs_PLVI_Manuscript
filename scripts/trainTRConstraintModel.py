### import modules
import torch
import pandas
import json
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn
from torch.nn import functional as F
import lightning as L
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
import os
import scipy.stats
from lightning.pytorch.callbacks import TQDMProgressBar, ModelCheckpoint

### set optional parameters -- any of these can be commented out
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision('medium') # ~15% speed boost if using a GPU with Tensor Cores

### load the data
batchSize=16
trainingSet=pandas.read_csv('training_10kbFromGenes.bed',sep='\t')
validationSet=pandas.read_csv('validation_10kbFromGenes.bed',sep='\t')
trainingSeqs=pandas.read_csv('training_10kbFromGenes_seqs_75bp.txt',sep='\t',low_memory=False,header=None,names=['TRID','seq'])
validationSeqs=pandas.read_csv('validation_10kbFromGenes_seqs_75bp.txt',sep='\t',low_memory=False,header=None,names=['TRID','seq'])
trainingSet=trainingSet.merge(trainingSeqs,how='inner',on='TRID')
validationSet=validationSet.merge(validationSeqs,how='inner',on='TRID')

# the stats in this next section were determined by running 'quantile' command in pandas on the trainingSet for the given column
# drop rows where the target value is greater than the 99.9th percentile for motif variance (0.250)
trainingSet=trainingSet.loc[trainingSet['lpsMotifVariability']<=0.250,:].reset_index(drop=True)
validationSet=validationSet.loc[validationSet['lpsMotifVariability']<=0.250,:].reset_index(drop=True)

# drop rows where the target value is greater than the 99.9th percentile for motif variance (4)
trainingSet=trainingSet.loc[trainingSet['numUniqueMotifs']<=4,:].reset_index(drop=True)
validationSet=validationSet.loc[validationSet['numUniqueMotifs']<=4,:].reset_index(drop=True)

# drop rows where the target value is greater than the 99th percentile for length variance (8.65)
trainingSet=trainingSet.loc[trainingSet['mclpsmStdev']<=8.65,:].reset_index(drop=True)
validationSet=validationSet.loc[validationSet['mclpsmStdev']<=8.65,:].reset_index(drop=True)

# drop rows where the target val is greater than the 99th percentile (42.42)
trainingSet=trainingSet.loc[trainingSet['totalLengthStdev']<=42.42,:].reset_index(drop=True)
validationSet=validationSet.loc[validationSet['totalLengthStdev']<=42.42,:].reset_index(drop=True)

# drop rows where the target val is greater than the 99th percentile (9.10)
trainingSet=trainingSet.loc[trainingSet['combinedLPSStdev']<=9.10,:].reset_index(drop=True)
validationSet=validationSet.loc[validationSet['combinedLPSStdev']<=9.10,:].reset_index(drop=True)

# drop rows where the target value is greater than the 99.9th percentile for motif variance (0.331)
trainingSet=trainingSet.loc[trainingSet['CPS']<=0.331,:].reset_index(drop=True)
validationSet=validationSet.loc[validationSet['CPS']<=0.331,:].reset_index(drop=True)

# standardize all scores
mclpsmStdev_trainingMean=trainingSet.loc[:,'mclpsmStdev'].mean()
mclpsmStdev_trainingStdev=trainingSet.loc[:,'mclpsmStdev'].std()
lpsMotifVariability_trainingMean=trainingSet.loc[:,'lpsMotifVariability'].mean()
lpsMotifVariability_trainingStdev=trainingSet.loc[:,'lpsMotifVariability'].std()
numUniqueMotifs_trainingMean=trainingSet.loc[:,'numUniqueMotifs'].mean()
numUniqueMotifs_trainingStdev=trainingSet.loc[:,'numUniqueMotifs'].std()
totalLengthStdev_trainingMean=trainingSet.loc[:,'totalLengthStdev'].mean()
totalLengthStdev_trainingStdev=trainingSet.loc[:,'totalLengthStdev'].std()
combinedLPSStdev_trainingMean=trainingSet.loc[:,'combinedLPSStdev'].mean()
combinedLPSStdev_trainingStdev=trainingSet.loc[:,'combinedLPSStdev'].std()
CPS_trainingMean=trainingSet.loc[:,'CPS'].mean()
CPS_trainingStdev=trainingSet.loc[:,'CPS'].std()
trainingSet['mclpsmStdev']=(trainingSet.loc[:,'mclpsmStdev'].values-mclpsmStdev_trainingMean)/mclpsmStdev_trainingStdev
trainingSet['lpsMotifVariability']=(trainingSet.loc[:,'lpsMotifVariability'].values-lpsMotifVariability_trainingMean)/lpsMotifVariability_trainingStdev
trainingSet['numUniqueMotifs']=(trainingSet.loc[:,'numUniqueMotifs'].values-numUniqueMotifs_trainingMean)/numUniqueMotifs_trainingStdev
trainingSet['totalLengthStdev']=(trainingSet.loc[:,'totalLengthStdev'].values-totalLengthStdev_trainingMean)/totalLengthStdev_trainingStdev
trainingSet['combinedLPSStdev']=(trainingSet.loc[:,'combinedLPSStdev'].values-combinedLPSStdev_trainingMean)/combinedLPSStdev_trainingStdev
trainingSet['CPS']=(trainingSet.loc[:,'CPS'].values-CPS_trainingMean)/CPS_trainingStdev
validationSet['mclpsmStdev']=(validationSet.loc[:,'mclpsmStdev'].values-mclpsmStdev_trainingMean)/mclpsmStdev_trainingStdev
validationSet['lpsMotifVariability']=(validationSet.loc[:,'lpsMotifVariability'].values-lpsMotifVariability_trainingMean)/lpsMotifVariability_trainingStdev
validationSet['numUniqueMotifs']=(validationSet.loc[:,'numUniqueMotifs'].values-numUniqueMotifs_trainingMean)/numUniqueMotifs_trainingStdev
validationSet['totalLengthStdev']=(validationSet.loc[:,'totalLengthStdev'].values-totalLengthStdev_trainingMean)/totalLengthStdev_trainingStdev
validationSet['combinedLPSStdev']=(validationSet.loc[:,'combinedLPSStdev'].values-combinedLPSStdev_trainingMean)/combinedLPSStdev_trainingStdev
validationSet['CPS']=(validationSet.loc[:,'CPS'].values-CPS_trainingMean)/CPS_trainingStdev

# save normalization stats so the trained model can be used for inference later
normStats = {
	'mclpsmStdev_mean': float(mclpsmStdev_trainingMean), 'mclpsmStdev_std': float(mclpsmStdev_trainingStdev),
	'lpsMotifVariability_mean': float(lpsMotifVariability_trainingMean), 'lpsMotifVariability_std': float(lpsMotifVariability_trainingStdev),
	'numUniqueMotifs_mean': float(numUniqueMotifs_trainingMean), 'numUniqueMotifs_std': float(numUniqueMotifs_trainingStdev),
	'totalLengthStdev_mean': float(totalLengthStdev_trainingMean), 'totalLengthStdev_std': float(totalLengthStdev_trainingStdev),
	'combinedLPSStdev_mean': float(combinedLPSStdev_trainingMean), 'combinedLPSStdev_std': float(combinedLPSStdev_trainingStdev),
	'CPS_mean': float(CPS_trainingMean), 'CPS_std': float(CPS_trainingStdev),
}
with open('normalization_stats.json', 'w') as f:
	json.dump(normStats, f, indent=2)


### initialize tokenizer (done once here so datasets can pre-tokenize on CPU workers)
tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/NTv3_650M_pre", trust_remote_code=True)

### set up data loader class
class DNASeqDataset(Dataset):
	def __init__(self, tokenizer, seq, target1, target2, target3, target4, target5, target6):
		batch_encoding = tokenizer(list(seq), add_special_tokens=False)
		self.input_ids = [torch.tensor(ids) for ids in batch_encoding['input_ids']]
		self.target1 = torch.tensor(target1.astype('float32'))
		self.target2 = torch.tensor(target2.astype('float32'))
		self.target3 = torch.tensor(target3.astype('float32'))
		self.target4 = torch.tensor(target4.astype('float32'))
		self.target5 = torch.tensor(target5.astype('float32'))
		self.target6 = torch.tensor(target6.astype('float32'))
	def __getitem__(self, index):
		return {'input_ids': self.input_ids[index], 'target1': self.target1[index], 'target2': self.target2[index], 'target3': self.target3[index], 'target4': self.target4[index], 'target5': self.target5[index], 'target6': self.target6[index]}
	def __len__(self):
		return len(self.target1)

### collate function: pads each batch to the longest sequence, rounded up to the next even number
def make_collate_fn(pad_token_id):
	def collate_fn(batch):
		input_ids_list = [item['input_ids'] for item in batch]
		max_len = max(ids.shape[0] for ids in input_ids_list)
		padded_len = (max_len + 127) // 128 * 128  # round up to next multiple of 128
		padded_ids = torch.full((len(batch), padded_len), pad_token_id, dtype=torch.long)
		for i, ids in enumerate(input_ids_list):
			padded_ids[i, :ids.shape[0]] = ids
		return {
			'input_ids': padded_ids,
			'attention_mask': (padded_ids != pad_token_id),
			'target1': torch.stack([item['target1'] for item in batch]),
			'target2': torch.stack([item['target2'] for item in batch]),
			'target3': torch.stack([item['target3'] for item in batch]),
			'target4': torch.stack([item['target4'] for item in batch]),
			'target5': torch.stack([item['target5'] for item in batch]),
			'target6': torch.stack([item['target6'] for item in batch]),
		}
	return collate_fn

### create training and validation data loaders
collate_fn = make_collate_fn(tokenizer.pad_token_id)
trainingDataset=DNASeqDataset(tokenizer, trainingSet.loc[:,'seq'].to_list(),trainingSet.loc[:,'mclpsmStdev'].values,trainingSet.loc[:,'lpsMotifVariability'].values,trainingSet.loc[:,'numUniqueMotifs'].values,trainingSet.loc[:,'totalLengthStdev'].values,trainingSet.loc[:,'combinedLPSStdev'].values,trainingSet.loc[:,'CPS'].values)
trainingLoader=DataLoader(trainingDataset, batch_size=batchSize, shuffle=True, num_workers=4, collate_fn=collate_fn)
validationDataset=DNASeqDataset(tokenizer, validationSet.loc[:,'seq'].to_list(),validationSet.loc[:,'mclpsmStdev'].values,validationSet.loc[:,'lpsMotifVariability'].values,validationSet.loc[:,'numUniqueMotifs'].values,validationSet.loc[:,'totalLengthStdev'].values,validationSet.loc[:,'combinedLPSStdev'].values,validationSet.loc[:,'CPS'].values)
validationLoader=DataLoader(validationDataset, batch_size=batchSize, shuffle=False, num_workers=4, collate_fn=collate_fn)


### Construct the classifier architecture as a Pytorch Lightning module
class trVariationModel(L.LightningModule):
	def __init__(self):
		super().__init__()
		# initialize NT3
		self.nt = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/NTv3_650M_pre", trust_remote_code=True)
		# output layer for length variation of longest pure segment with the most common motif
		self.dense1 = torch.nn.Linear(1536,1)
		# output layer for motif variation
		self.dense2 = torch.nn.Linear(1536,1)
		# output layer for number of unique LPS motifs
		self.dense3 = torch.nn.Linear(1536,1)
		# output layer for overall length variation
		self.dense4 = torch.nn.Linear(1536,1)
		# output layer for length variation of longest pure segment with any motif
		self.dense5 = torch.nn.Linear(1536,1)
		# output layer for motif variation (CPS)
		self.dense6 = torch.nn.Linear(1536,1)
		# attention pooling layer
		self.attn = torch.nn.Linear(1536,1)
		self.val_y_len1=torch.FloatTensor()
		self.val_y_motif1=torch.FloatTensor()
		self.val_y_motif2=torch.FloatTensor()
		self.val_y_len2=torch.FloatTensor()
		self.val_y_len3=torch.FloatTensor()
		self.val_y_motif3=torch.FloatTensor()
		self.val_pred_len1=torch.FloatTensor()
		self.val_pred_motif1=torch.FloatTensor()
		self.val_pred_motif2=torch.FloatTensor()
		self.val_pred_len2=torch.FloatTensor()
		self.val_pred_len3=torch.FloatTensor()
		self.val_pred_motif3=torch.FloatTensor()

	def forward(self, dataIn):
		# pass pre-tokenized tensors directly through NucleotideTransformer
		seq = self.nt(dataIn['input_ids'], attention_mask=dataIn['attention_mask'], output_hidden_states=True)
		# attention-weighted mean pooling
		hidden = seq['hidden_states'][-1]                                                    # (batch, seq_len, 1536)
		attn_scores = self.attn(hidden).squeeze(-1)                                          # (batch, seq_len)
		attn_scores = attn_scores.masked_fill(~dataIn['attention_mask'], float('-inf'))
		attn_weights = torch.softmax(attn_scores, dim=1)                                     # (batch, seq_len)
		X = (attn_weights.unsqueeze(-1) * hidden).sum(dim=1)                                 # (batch, 1536)
		y1 = self.dense1(X)
		y2 = self.dense2(X)
		y3 = self.dense3(X)
		y4 = self.dense4(X)
		y5 = self.dense5(X)
		y6 = self.dense6(X)
		return y1, y2, y3, y4, y5, y6

	def configure_optimizers(self):
		optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5, weight_decay=0.01)
		scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4, total_steps=self.trainer.estimated_stepping_batches)
		return {'optimizer':optimizer,'lr_scheduler':{'scheduler':scheduler,'interval':'step'}}

	def training_step(self, train_batch, batch_idx):
		y1 = train_batch['target1']
		y2 = train_batch['target2']
		y3 = train_batch['target3']
		y4 = train_batch['target4']
		y5 = train_batch['target5']
		y6 = train_batch['target6']
		preds1, preds2, preds3, preds4, preds5, preds6 = self(train_batch)
		preds1 = torch.squeeze(preds1, dim=1)
		preds2 = torch.squeeze(preds2, dim=1)
		preds3 = torch.squeeze(preds3, dim=1)
		preds4 = torch.squeeze(preds4, dim=1)
		preds5 = torch.squeeze(preds5, dim=1)
		preds6 = torch.squeeze(preds6, dim=1)
		loss1 = F.mse_loss(preds1, y1)
		loss2 = F.mse_loss(preds2, y2)
		loss3 = F.mse_loss(preds3, y3)
		loss4 = F.mse_loss(preds4, y4)
		loss5 = F.mse_loss(preds5, y5)
		loss6 = F.mse_loss(preds6, y6)
		loss = loss1 + (loss2) + loss3 + (loss4) + loss5 + (loss6)
		self.log('train_loss_len1', loss1, on_step=True, on_epoch=False, prog_bar=True)
		self.log('train_loss_len2', loss4, on_step=True, on_epoch=False, prog_bar=True)
		self.log('train_loss_len3', loss5, on_step=True, on_epoch=False, prog_bar=True)
		self.log('train_loss_motif1', loss2, on_step=True, on_epoch=False, prog_bar=True)
		self.log('train_loss_motif2', loss3, on_step=True, on_epoch=False, prog_bar=True)
		self.log('train_loss_motif3', loss6, on_step=True, on_epoch=False, prog_bar=True)
		return loss

	def validation_step(self, val_batch, batch_idx):
		y1 = val_batch['target1']
		y2 = val_batch['target2']
		y3 = val_batch['target3']
		y4 = val_batch['target4']
		y5 = val_batch['target5']
		y6 = val_batch['target6']
		preds1, preds2, preds3, preds4, preds5, preds6 = self(val_batch)
		preds1 = torch.squeeze(preds1, dim=1)
		preds2 = torch.squeeze(preds2, dim=1)
		preds3 = torch.squeeze(preds3, dim=1)
		preds4 = torch.squeeze(preds4, dim=1)
		preds5 = torch.squeeze(preds5, dim=1)
		preds6 = torch.squeeze(preds6, dim=1)
		loss1 = F.mse_loss(preds1, y1)
		loss2 = F.mse_loss(preds2, y2)
		loss3 = F.mse_loss(preds3, y3)
		loss4 = F.mse_loss(preds4, y4)
		loss5 = F.mse_loss(preds5, y5)
		loss6 = F.mse_loss(preds6, y6)
		loss = loss1 + (loss2) + loss3 + (loss4) + loss5 + (loss6)
		self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
		self.val_y_len1=torch.cat([self.val_y_len1,y1.cpu()])
		self.val_y_motif1=torch.cat([self.val_y_motif1,y2.cpu()])
		self.val_y_motif2=torch.cat([self.val_y_motif2,y3.cpu()])
		self.val_y_len2=torch.cat([self.val_y_len2,y4.cpu()])
		self.val_y_len3=torch.cat([self.val_y_len3,y5.cpu()])
		self.val_y_motif3=torch.cat([self.val_y_motif3,y6.cpu()])
		self.val_pred_len1=torch.cat([self.val_pred_len1,preds1.cpu()])
		self.val_pred_motif1=torch.cat([self.val_pred_motif1,preds2.cpu()])
		self.val_pred_motif2=torch.cat([self.val_pred_motif2,preds3.cpu()])
		self.val_pred_len2=torch.cat([self.val_pred_len2,preds4.cpu()])
		self.val_pred_len3=torch.cat([self.val_pred_len3,preds5.cpu()])
		self.val_pred_motif3=torch.cat([self.val_pred_motif3,preds6.cpu()])

	def test_step(self, test_batch, batch_idx):
		y1 = test_batch['target1']
		y2 = test_batch['target2']
		y3 = test_batch['target3']
		y4 = test_batch['target4']
		y5 = test_batch['target5']
		y6 = test_batch['target6']
		preds1, preds2, preds3, preds4, preds5, preds6 = self(test_batch)
		preds1 = torch.squeeze(preds1, dim=1)
		preds2 = torch.squeeze(preds2, dim=1)
		preds3 = torch.squeeze(preds3, dim=1)
		preds4 = torch.squeeze(preds4, dim=1)
		preds5 = torch.squeeze(preds5, dim=1)
		preds6 = torch.squeeze(preds6, dim=1)
		loss1 = F.mse_loss(preds1, y1)
		loss2 = F.mse_loss(preds2, y2)
		loss3 = F.mse_loss(preds3, y3)
		loss4 = F.mse_loss(preds4, y4)
		loss5 = F.mse_loss(preds5, y5)
		loss6 = F.mse_loss(preds6, y6)
		loss = loss1 + (loss2) + loss3 + (loss4) + loss5 + (loss6)


	def on_validation_epoch_end(self):
		print("Validation set length performance (mclpsmStdev)")
		print(scipy.stats.pearsonr(self.val_y_len1, self.val_pred_len1))
		print(scipy.stats.spearmanr(self.val_y_len1, self.val_pred_len1))
		print("")
		print("Validation set motif performance (lpsMotifVariability)")
		print(scipy.stats.pearsonr(self.val_y_motif1, self.val_pred_motif1))
		print(scipy.stats.spearmanr(self.val_y_motif1, self.val_pred_motif1))
		print("")
		print("Validation set motif performance (numUniqueMotifs)")
		print(scipy.stats.pearsonr(self.val_y_motif2, self.val_pred_motif2))
		print(scipy.stats.spearmanr(self.val_y_motif2, self.val_pred_motif2))
		print("")
		print("Validation set length performance (totalLengthStdev)")
		print(scipy.stats.pearsonr(self.val_y_len2, self.val_pred_len2))
		print(scipy.stats.spearmanr(self.val_y_len2, self.val_pred_len2))
		print("")
		print("Validation set length performance (combinedLPSStdev)")
		print(scipy.stats.pearsonr(self.val_y_len3, self.val_pred_len3))
		print(scipy.stats.spearmanr(self.val_y_len3, self.val_pred_len3))
		print("")
		print("Validation set motif performance (CPS)")
		print(scipy.stats.pearsonr(self.val_y_motif3, self.val_pred_motif3))
		print(scipy.stats.spearmanr(self.val_y_motif3, self.val_pred_motif3))
		print("")
		self.val_y_len1=torch.FloatTensor()
		self.val_y_motif1=torch.FloatTensor()
		self.val_y_motif2=torch.FloatTensor()
		self.val_y_motif3=torch.FloatTensor()
		self.val_y_len2=torch.FloatTensor()
		self.val_y_len3=torch.FloatTensor()
		self.val_pred_len1=torch.FloatTensor()
		self.val_pred_motif1=torch.FloatTensor()
		self.val_pred_motif2=torch.FloatTensor()
		self.val_pred_motif3=torch.FloatTensor()
		self.val_pred_len2=torch.FloatTensor()
		self.val_pred_len3=torch.FloatTensor()

# make the progress bar not overwrite itself every epoch
class LitProgressBar(TQDMProgressBar):
	def on_train_epoch_end(self, *args, **kwargs):
		super().on_train_epoch_end(*args, **kwargs)
		print()
progBar=LitProgressBar()


### Instantiate the model
#model = trVariationModel()
model = trVariationModel.load_from_checkpoint('models/best-epoch=07-val_loss=3.7290.ckpt', strict=False)

### Train the model
grad_batches=int(512 / batchSize)
checkpoint_cb = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=3, save_last=True, dirpath='models', filename='best-{epoch:02d}-{val_loss:.4f}')
trainer = L.Trainer(accelerator='gpu', callbacks=[progBar, checkpoint_cb], max_epochs=10, precision='bf16-mixed', gradient_clip_val=1.0, accumulate_grad_batches=grad_batches)
#trainer.fit(model, trainingLoader, validationLoader)


### Test the model using the validation set
y_valid_pred=trainer.predict(model,validationLoader)
y_valid_pred2_len1=torch.cat([b[0] for b in y_valid_pred]).squeeze().cpu().float().numpy()
y_valid_pred2_motif1=torch.cat([b[1] for b in y_valid_pred]).squeeze().cpu().float().numpy()
y_valid_pred2_motif2=torch.cat([b[2] for b in y_valid_pred]).squeeze().cpu().float().numpy()
y_valid_pred2_len2=torch.cat([b[3] for b in y_valid_pred]).squeeze().cpu().float().numpy()
y_valid_pred2_len3=torch.cat([b[4] for b in y_valid_pred]).squeeze().cpu().float().numpy()
y_valid_pred2_motif3=torch.cat([b[5] for b in y_valid_pred]).squeeze().cpu().float().numpy()
print("Validation set length performance (mclpsmStdev)")
print(scipy.stats.pearsonr(validationSet.loc[:,'mclpsmStdev'].values, y_valid_pred2_len1))
print(scipy.stats.spearmanr(validationSet.loc[:,'mclpsmStdev'].values, y_valid_pred2_len1))
print("")
print("Validation set motif performance (lpsMotifVariability)")
print(scipy.stats.pearsonr(validationSet.loc[:,'lpsMotifVariability'].values, y_valid_pred2_motif1))
print(scipy.stats.spearmanr(validationSet.loc[:,'lpsMotifVariability'].values, y_valid_pred2_motif1))
print("")
print("Validation set motif performance (numUniqueMotifs)")
print(scipy.stats.pearsonr(validationSet.loc[:,'numUniqueMotifs'].values, y_valid_pred2_motif2))
print(scipy.stats.spearmanr(validationSet.loc[:,'numUniqueMotifs'].values, y_valid_pred2_motif2))
print("")
print("Validation set length performance (totalLengthStdev)")
print(scipy.stats.pearsonr(validationSet.loc[:,'totalLengthStdev'].values, y_valid_pred2_len2))
print(scipy.stats.spearmanr(validationSet.loc[:,'totalLengthStdev'].values, y_valid_pred2_len2))
print("")
print("Validation set length performance (combinedLPSStdev)")
print(scipy.stats.pearsonr(validationSet.loc[:,'combinedLPSStdev'].values, y_valid_pred2_len3))
print(scipy.stats.spearmanr(validationSet.loc[:,'combinedLPSStdev'].values, y_valid_pred2_len3))
print("")
print("Validation set motif performance (CPS)")
print(scipy.stats.pearsonr(validationSet.loc[:,'CPS'].values, y_valid_pred2_motif3))
print(scipy.stats.spearmanr(validationSet.loc[:,'CPS'].values, y_valid_pred2_motif3))


# Make predictions on the genic set 
testSet=pandas.read_csv('testing_within10kbOfGenes.bed',sep='\t')
testSeqs=pandas.read_csv('testing_within10kbOfGenes_seqs_75bp.txt',sep='\t',low_memory=False,header=None,names=['TRID','seq'])
testSet=testSet.merge(testSeqs,how='inner',on='TRID')

# standardize testSet
testSet['mclpsmStdev']=(testSet.loc[:,'mclpsmStdev'].values-mclpsmStdev_trainingMean)/mclpsmStdev_trainingStdev
testSet['lpsMotifVariability']=(testSet.loc[:,'lpsMotifVariability'].values-lpsMotifVariability_trainingMean)/lpsMotifVariability_trainingStdev
testSet['numUniqueMotifs']=(testSet.loc[:,'numUniqueMotifs'].values-numUniqueMotifs_trainingMean)/numUniqueMotifs_trainingStdev
testSet['totalLengthStdev']=(testSet.loc[:,'totalLengthStdev'].values-totalLengthStdev_trainingMean)/totalLengthStdev_trainingStdev
testSet['combinedLPSStdev']=(testSet.loc[:,'combinedLPSStdev'].values-combinedLPSStdev_trainingMean)/combinedLPSStdev_trainingStdev
testSet['CPS']=(testSet.loc[:,'CPS'].values-CPS_trainingMean)/CPS_trainingStdev


testDataset=DNASeqDataset(tokenizer, testSet.loc[:,'seq'].to_list(),testSet.loc[:,'mclpsmStdev'].values,testSet.loc[:,'lpsMotifVariability'].values,testSet.loc[:,'numUniqueMotifs'].values,testSet.loc[:,'totalLengthStdev'].values,testSet.loc[:,'combinedLPSStdev'].values,testSet.loc[:,'CPS'].values)
testLoader=DataLoader(testDataset, batch_size=batchSize, shuffle=False, num_workers=4, collate_fn=collate_fn)

y_pred=trainer.predict(model,testLoader)
y_pred2_len1=torch.cat([b[0] for b in y_pred]).squeeze().cpu().float().numpy()
y_pred2_motif1=torch.cat([b[1] for b in y_pred]).squeeze().cpu().float().numpy()
y_pred2_motif2=torch.cat([b[2] for b in y_pred]).squeeze().cpu().float().numpy()
y_pred2_len2=torch.cat([b[3] for b in y_pred]).squeeze().cpu().float().numpy()
y_pred2_len3=torch.cat([b[4] for b in y_pred]).squeeze().cpu().float().numpy()
y_pred2_motif3=torch.cat([b[5] for b in y_pred]).squeeze().cpu().float().numpy()
testSet['expectedMCLPSStdev']=y_pred2_len1
testSet['expectedLPSMotifVariability']=y_pred2_motif1
testSet['expectedNumUniqueMotifs']=y_pred2_motif2
testSet['expectedTotalLengthStdev']=y_pred2_len2
testSet['expectedCombinedLPSStdev']=y_pred2_len3
testSet['expectedCPS']=y_pred2_motif3

# convert the standardized scores back to their original values
testSet['mclpsmStdev']=(testSet.loc[:,'mclpsmStdev'].values*mclpsmStdev_trainingStdev)+mclpsmStdev_trainingMean
testSet['lpsMotifVariability']=(testSet.loc[:,'lpsMotifVariability'].values*lpsMotifVariability_trainingStdev)+lpsMotifVariability_trainingMean
testSet['numUniqueMotifs']=(testSet.loc[:,'numUniqueMotifs'].values*numUniqueMotifs_trainingStdev)+numUniqueMotifs_trainingMean
testSet['totalLengthStdev']=(testSet.loc[:,'totalLengthStdev'].values*totalLengthStdev_trainingStdev)+totalLengthStdev_trainingMean
testSet['combinedLPSStdev']=(testSet.loc[:,'combinedLPSStdev'].values*combinedLPSStdev_trainingStdev)+combinedLPSStdev_trainingMean
testSet['CPS']=(testSet.loc[:,'CPS'].values*CPS_trainingStdev)+CPS_trainingMean
testSet['expectedMCLPSStdev']=(testSet.loc[:,'expectedMCLPSStdev'].values*mclpsmStdev_trainingStdev)+mclpsmStdev_trainingMean
testSet['expectedLPSMotifVariability']=(testSet.loc[:,'expectedLPSMotifVariability'].values*lpsMotifVariability_trainingStdev)+lpsMotifVariability_trainingMean
testSet['expectedNumUniqueMotifs']=(testSet.loc[:,'expectedNumUniqueMotifs'].values*numUniqueMotifs_trainingStdev)+numUniqueMotifs_trainingMean
testSet['expectedTotalLengthStdev']=(testSet.loc[:,'expectedTotalLengthStdev'].values*totalLengthStdev_trainingStdev)+totalLengthStdev_trainingMean
testSet['expectedCombinedLPSStdev']=(testSet.loc[:,'expectedCombinedLPSStdev'].values*combinedLPSStdev_trainingStdev)+combinedLPSStdev_trainingMean
testSet['expectedCPS']=(testSet.loc[:,'expectedCPS'].values*CPS_trainingStdev)+CPS_trainingMean

testSet=testSet.drop(columns='seq')
testSet.to_csv('testing_within10kbOfGenes_scoredNT3.bed',sep='\t',index=False)
