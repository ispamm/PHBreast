import torch
import time
import torch.nn as nn
import wandb
import sys
from models.hypercomplex_layers import PHConv
from sklearn.metrics import roc_auc_score
sys.path.append('early-stopping-pytorch')
from pytorchtools import EarlyStopping
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

class Trainer():
    def __init__(self, net, optimizer, epochs,
                      use_cuda=True, gpu_num=0,
                      checkpoint_folder="./checkpoints",
                      l1_reg=False,
                      num_classes=1,
                      num_views=2,
                      pos_weight=None,
                      distributed=False,
                      rank=0,
                      world_size=None):

        self.optimizer = optimizer
        self.epochs = epochs
        self.use_cuda = use_cuda
        self.gpu_num = gpu_num
        self.checkpoints_folder = checkpoint_folder
        self.l1_reg = l1_reg
        self.rank = rank
        self.distributed = distributed
        self.world_size = world_size
        self.num_classes = num_classes
        self.num_views = num_views

        if num_classes == 1:
            pos_weight = torch.tensor([pos_weight]) if pos_weight else None
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            self.val_criterion = nn.BCEWithLogitsLoss()

        else:
            self.criterion = nn.CrossEntropyLoss()
            self.val_criterion = self.criterion
        
        if self.use_cuda:
            if pos_weight:
                self.criterion.pos_weight = torch.tensor([pos_weight]).cuda('cuda:%i' %self.gpu_num)

            print(f"[Proc{rank}]Running on GPU?", self.use_cuda, "- gpu_num: ", self.gpu_num)
            self.net = net.cuda('cuda:%i' %self.gpu_num)
            
            if distributed:
                self.net = DDP(self.net, device_ids=[self.gpu_num], output_device=self.gpu_num, find_unused_parameters=True)
        else:
            self.net = net

    def train(self, train_loader, eval_loader):
        
        # name for checkpoint
        run_name = wandb.run.name if self.rank==0 else None

        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=20, path=self.checkpoints_folder + "/best_" + run_name + ".pt", rank=self.rank)
        
        for epoch in range(self.epochs):  # loop over the dataset multiple times
            
            if self.distributed:
                train_loader.sampler.set_epoch(epoch)  

            start = time.time()
            running_loss_train = 0.0
            running_loss_eval = 0.0
            total = 0.0
            correct = 0.0
            y_pred = torch.empty(0)
            y_true = torch.empty(0)
           
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data

                if self.num_views == 4:
                    labels = torch.cat([labels[0], labels[1]], dim=0)
                
                if self.num_classes == 1:
                    labels = labels.view((-1, 1)).to(torch.float32)

                if self.use_cuda:
                    inputs, labels = inputs.cuda('cuda:%i' %self.gpu_num), labels.cuda('cuda:%i' %self.gpu_num)
                    
                self.optimizer.zero_grad()
                
                if self.num_views == 4:
                    inputs = torch.split(inputs, split_size_or_sections=2, dim=1)

                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)

                if self.l1_reg:
                    # Add L1 regularization to A
                    regularization_loss = 0.0
                    for child in self.net.children():
                        for layer in child.modules():
                            if isinstance(layer, PHConv):
                                for param in layer.a:
                                    regularization_loss += torch.sum(abs(param))
                    loss += 0.001 * regularization_loss


                loss.backward()
                self.optimizer.step()

                running_loss_train += loss.item()
                
            end = time.time()
            
           
            self.net.eval()
            
            if self.distributed:
                to_gather = dict(y_pred=None, y_true=None, loss_eval=None, loss_train=running_loss_train)
                
            for j, eval_data in enumerate(eval_loader, 0):
                inputs, labels = eval_data

                if self.num_views == 4:
                    labels = torch.cat([labels[0], labels[1]], dim=0)
                    
                if self.num_classes == 1:
                    labels = labels.view((-1, 1)).to(torch.float32)

                if self.use_cuda:
                    inputs, labels = inputs.cuda('cuda:%i' %self.gpu_num), labels.cuda('cuda:%i' %self.gpu_num)
                
                if self.num_views == 4:
                    inputs = torch.split(inputs, split_size_or_sections=2, dim=1)
                    
                eval_outputs = self.net(inputs)
                eval_loss = self.val_criterion(eval_outputs, labels)
                running_loss_eval += eval_loss.item()

                # for multi-class (patch)
                if self.num_classes == 1:
                    predicted = torch.sigmoid(eval_outputs) > 0.5
                else:
                    _, predicted = torch.max(eval_outputs.data, 1)  
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                acc = 100*correct/total

                y_pred = torch.cat((y_pred, predicted.view(predicted.shape[0]).cpu()))
                y_true = torch.cat((y_true, labels.view(labels.shape[0]).cpu()))

            
            if self.distributed:
                to_gather["y_pred"] = y_pred
                to_gather["y_true"] = y_true
                to_gather["loss_eval"] = running_loss_eval
                gathered = [None for _ in range(self.world_size)]
                
                dist.all_gather_object(gathered, to_gather)
                
                # NB: Not implemented for world_size > 2 
                y_pred = torch.cat((gathered[0]["y_pred"], gathered[1]["y_pred"]))
                y_true = torch.cat((gathered[0]["y_true"], gathered[1]["y_true"]))

                total = y_true.shape[0]
                correct = (y_pred == y_true).sum().item()
                acc = 100*correct/total

                if self.num_classes == 1:
                    auc = roc_auc_score(y_true, y_pred)
                
                running_loss_train = gathered[0]["loss_train"] + gathered[1]["loss_train"]
                running_loss_eval = gathered[0]["loss_eval"] + gathered[1]["loss_eval"]
                
                i *= 2
                j *= 2
                
            elif self.num_classes == 1:
                auc = roc_auc_score(y_true, y_pred)

            # Log metrics
            if self.rank == 0:
                wandb.log({"train loss": running_loss_train/i, "epoch": epoch+1})
                wandb.log({"val loss": running_loss_eval/j, "epoch": epoch+1})
                wandb.log({"val acc": acc, "epoch": epoch+1})

                if self.num_classes == 1:
                    wandb.log({"val auc": auc, "epoch": epoch+1})
                    print("[Epoch: %i][Train Loss: %f][Val Loss: %f][Val Acc: %f][Val AUC: %f][Time: %f]" %(epoch+1, running_loss_train/i, running_loss_eval/j, acc, auc, end-start))
                else:
                    print("[Epoch: %i][Train Loss: %f][Val Loss: %f][Val Acc: %f][Time: %f]" %(epoch+1, running_loss_train/i, running_loss_eval/j, acc, end-start))
            
            # Early stopping
            if self.num_classes == 1:
                early_stopping(auc, self.net)
            else:
                early_stopping(acc, self.net)

            if early_stopping.early_stop:
                print(f"Proc[{self.rank}]Early stopping")
                break
            
            running_loss_train = 0.0
            running_loss_eval = 0.0
            self.net.train()
            

        if self.distributed and self.rank == 0:
            wandb.finish()
            
        print(f'[Proc{self.rank}]Finished Training')
    
    def test(self, test_loader):
        print("Testing net...")
        
        for name, params in self.net.named_parameters():
            params.requires_grad = False

        self.net.eval()

        if self.use_cuda:
            self.net = self.net.cuda('cuda:%i' %self.gpu_num)

        correct = 0.0
        total = 0.0
        y_pred = torch.empty(0)
        y_true = torch.empty(0)

        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data    

                if self.num_views == 4:              
                    labels = torch.cat([labels[0], labels[1]], dim=0)
                    
                if self.num_classes == 1:
                    labels = labels.view((-1, 1)).to(torch.float32)
    
                if self.use_cuda:
                    inputs, labels = inputs.cuda('cuda:%i' %self.gpu_num), labels.cuda('cuda:%i' %self.gpu_num)
                
                if self.num_views == 4:
                    inputs = torch.split(inputs, split_size_or_sections=2, dim=1)
                
                eval_outputs = self.net(inputs)

                if self.num_classes == 1:
                    predicted = torch.sigmoid(eval_outputs) > 0.5
                else: # for multi-class (patch)
                    _, predicted = torch.max(eval_outputs.data, 1)  
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                y_pred = torch.cat((y_pred, predicted.view(predicted.shape[0]).cpu()))
                y_true = torch.cat((y_true, labels.view(labels.shape[0]).cpu()))
        
        if self.num_classes == 1:
            auc = roc_auc_score(y_true, y_pred)
            print('AUC %s on the test images: %.3f' % (self.net.__class__.__name__, auc))
            wandb.log({"Test AUC": auc})
        
        print('Accuracy %s on the test images: %.3f %%' % (self.net.__class__.__name__, 100*correct/total))
        wandb.log({"Test Accuracy": 100*correct/total})
       
        
