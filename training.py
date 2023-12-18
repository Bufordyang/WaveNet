import argparse
from utils.preprocess_data import PreDataLoader
from utils.utils import random_planetoid_splits
from models import *
import torch
import torch.nn.functional as F
from tqdm import tqdm
import random
import seaborn as sns
import numpy as np
import time
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def RunExp(args, dataset, data, Net, percls_trn, val_lb):

    def train(model, optimizer, data, dprate):
        
        model.train()
        optimizer.zero_grad()
        out = model(data)[data.train_mask]
        nll = F.nll_loss(out, data.y[data.train_mask])
        loss = nll
        reg_loss=None
        loss.backward()
        optimizer.step()
        del out
        return loss.clone().detach().cpu()

    def test(model, data):
        model.eval()
        logits, accs, losses, preds = model(data), [], [], []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

            loss = F.nll_loss(model(data)[mask], data.y[mask])
            preds.append(pred.detach().cpu())
            accs.append(acc)
            losses.append(loss.detach().cpu())

        return accs, preds, losses

    device = torch.device('cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu')
    tmp_net = Net(dataset, args)

    #randomly split dataset
    data = random_planetoid_splits(data, dataset.num_classes, percls_trn, val_lb,args.seed)
    model, data = tmp_net.to(device), data.to(device)

    if args.net=='GPRGNN':
        optimizer = torch.optim.Adam([{ 'params': model.lin1.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
        {'params': model.lin2.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
        {'params': model.prop1.parameters(), 'weight_decay': 0.00, 'lr': args.lr}])

    elif args.net =='BernNet':
        optimizer = torch.optim.Adam([{'params': model.lin1.parameters(),'weight_decay': args.weight_decay, 'lr': args.lr},
        {'params': model.lin2.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
        {'params': model.prop1.parameters(), 'weight_decay': args.weight_decay, 'lr': args.prop_lr}])

    elif args.net == 'ChebNetII':
        optimizer = torch.optim.Adam([{ 'params': model.lin1.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
        {'params': model.lin2.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
        {'params': model.prop1.parameters(), 'weight_decay': args.prop_wd, 'lr': args.prop_lr}])
    
    elif args.net == 'WaveNet':
        if args.dataset in ['Toygraph_sin','Toygraph_esin']:
            optimizer = torch.optim.SGD([{'params': model.lin1.parameters(),'weight_decay': args.weight_decay, 'lr': args.lr},
            {'params': model.lin2.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
            {'params': model.prop1.parameters(), 'weight_decay': args.weight_decay, 'lr': args.prop_lr}])
        else:
            optimizer = torch.optim.Adam([{'params': model.lin1.parameters(),'weight_decay': args.weight_decay, 'lr': args.lr},
            {'params': model.lin2.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
            {'params': model.prop1.parameters(), 'weight_decay': args.weight_decay, 'lr': args.prop_lr}])
    else:
        optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)

    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []

    time_run=[]
    for epoch in range(args.epochs):
        t_st=time.time()
        train_loss = train(model, optimizer, data, args.dprate)

        time_epoch=time.time()-t_st  # each epoch train times
        time_run.append(time_epoch)

        [train_acc, val_acc, tmp_test_acc], preds, \
        [train_loss, val_loss, tmp_test_loss] = test(model, data)

        if val_loss < best_val_loss:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc
            if args.net in ['ChebNetII', 'BernNet','WaveNet']:
                TEST = tmp_net.prop1.temp.clone()
                theta = TEST.detach().cpu()
                if args.net in ['BernNet','ChebNetII']:
                    theta = torch.relu(theta).numpy()
                # print(tmp_net.prop1.temp.grad)
            else:
                theta = args.alpha

        if epoch >= 0:
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            if args.early_stopping > 0 and epoch > args.early_stopping:
                tmp = torch.tensor(
                    val_loss_history[-(args.early_stopping + 1):-1])
                if val_loss > tmp.mean().item():
                    print('The sum of epochs:',epoch)
                    break
    
    return test_acc, best_val_acc, theta, time_run



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='seeds for random splits.')
    parser.add_argument('--epochs', type=int, default=400, help='max epochs.')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')       
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay.')  
    parser.add_argument('--early_stopping', type=int, default=200, help='early stopping.')
    parser.add_argument('--hidden', type=int, default=64, help='hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout for neural networks.')

    parser.add_argument('--train_rate', type=float, default=0.6, help='train set rate.')
    parser.add_argument('--val_rate', type=float, default=0.2, help='val set rate.')
    parser.add_argument('--K', type=int, default=10, help='num of bases.')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha for APPN/GPRGNN.')
    parser.add_argument('--dprate', type=float, default=0.5, help='dropout for propagation layer.')
    parser.add_argument('--Init', type=str,choices=['SGC', 'PPR', 'NPPR', 'Random', 'WS', 'Null'], default='PPR', help='initialization for GPRGNN.')
    parser.add_argument('--heads', default=8, type=int, help='attention heads for GAT.')
    parser.add_argument('--output_heads', default=1, type=int, help='output_heads for GAT.')

    parser.add_argument('--dataset', type=str, choices=['Cora','Citeseer','Pubmed',
                                                        'Computers','Photo',
                                                        'Chameleon','Squirrel','Actor',
                                                        'Toygraph_sin','Toygraph_esin'],
                                                        default='Chameleon',help='Optional datasets name: Cora|Citeseer|Pubmed|Computers|Photo|Chameleon|Squirrel|Actor|Toygraph')
    parser.add_argument('--device', type=int, default=0, help='GPU device.')
    parser.add_argument('--runs', type=int, default=10, help='number of runs.')
    parser.add_argument('--net', type=str, 
                        choices=['GCN', 'GAT', 'APPNP', 
                                 'ChebNet', 'GPRGNN', 'BernNet',
                                 'MLP', 'ChebNetII', 'SplineCNN', 'WaveNet'], default='WaveNet')
    parser.add_argument('--prop_wd', type=float, default=0.0005, help='learning rate for propagation layer.')
    parser.add_argument('--prop_lr', type=float, default=0.01, help='learning rate for propagation layer.')

    parser.add_argument('--prefix',type=str, default='0', help='sample_prefix')

    args = parser.parse_args()
    set_seed(1)

    import random 
    SEEDS=[3280387012, 1095513148, 1930549411, 2798570523, 3387541014, 403123852, 3589583794, 1912923437, 4059906722, 3871601465]       # 10 runs fixed seed

    print(args)
    print("---------------------------------------------")

    gnn_name = args.net
    if gnn_name == 'GCN':
        Net = GCN_Net
    elif gnn_name == 'GAT':
        Net = GAT_Net
    elif gnn_name == 'APPNP':
        Net = APPNP_Net
    elif gnn_name == 'ChebNet':
        Net = ChebNet
    elif gnn_name =='ChebNetII':
        Net = ChebNetII
    elif gnn_name == 'GPRGNN':
        Net = GPRGNN
    elif gnn_name == 'BernNet':
        Net = BernNet
    elif gnn_name =='MLP':
        Net = MLP
    elif gnn_name =='SplineCNN':
        Net = SplineCNN
    elif gnn_name =='WaveNet':
        Net = WaveNet

    dataset = PreDataLoader(args.dataset,args.net, args.prefix)
    print(f'{args.dataset} loaded!')

    try:
        data = dataset.data
    except:
        data = dataset[0]

    
    percls_trn = int(round(args.train_rate*len(data.y)/dataset.num_classes))
    val_lb = int(round(args.val_rate*len(data.y)))


    results = []
    time_results=[]
    for RP in tqdm(range(args.runs)):
        args.seed=SEEDS[RP]
        test_acc, best_val_acc, theta_0,time_run = RunExp(args, dataset, data, Net, percls_trn, val_lb)
        time_results.append(time_run)
        results.append([test_acc, best_val_acc, theta_0])
        print(f'run_{str(RP+1)} \t test_acc: {test_acc:.4f}')
        if args.net in ['BernNet', 'WaveNet','WaveNet-S']:
            print('Theta:', [float('{:.4f}'.format(i)) for i in theta_0])

    run_sum=0
    epochsss=0
    for i in time_results:
        run_sum+=sum(i)
        epochsss+=len(i)

    print("each run avg_time:",run_sum/(args.runs),"s")
    print("each epoch avg_time:",1000*run_sum/epochsss,"ms")
    results_values = [[row[0], row[1]] for row in results]
    test_acc_mean, val_acc_mean = np.mean(results_values, axis=0) * 100
    test_acc_std = np.sqrt(np.var(results_values, axis=0)[0]) * 100

    values=np.asarray(results_values)[:,0]
    uncertainty=np.max(np.abs(sns.utils.ci(sns.algorithms.bootstrap(values,func=np.mean,n_boot=1000),95)-values.mean()))

    print(f'{gnn_name} on dataset {args.dataset}, in {args.runs} repeated experiment:')
    print(f'test acc mean = {test_acc_mean:.4f} ± {uncertainty*100:.4f}  \t val acc mean = {val_acc_mean:.4f}')
