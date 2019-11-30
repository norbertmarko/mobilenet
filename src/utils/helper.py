import sys
import os 

sys.path.append('../..')
import params


def printHyperparams():
    print("Warmup epochs: {}\n".format(params.epochs_wu) 
        + "Fine-tune epochs: {}\n".format(params.epochs_ft)
        + "Batch Size: {}\n".format(params.batch_size)
        + "Validation Batch Size: {}\n".format(params.val_batch_size)
        + "Weight Decay: {}\n".format(params.weight_decay)
        + "Warmup Learning Rate: {}\n".format(params.learning_rate_wu)
        + "Fine-tune Learning Rate: {}\n".format(params.learning_rate_ft)
        + "Warmup start layer: {}\n".format(params.warmupStartLayer)
        + "Fine-tune start layer: {}\n".format(params.finetuneStartLayer)
	+ "Augmentation: {}\n".format(params.augmentation))
    
if __name__ == "__main__":
    printHyperparams()
