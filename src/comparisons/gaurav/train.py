import umap

from src.comparisons.gaurav.model import *
from src.data.data import get_ds

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    class_to_index={"standing":0, "running":1, "walking":2,"badminton":3}
    train_ds, test_ds = get_ds("data/basicmotions/processed/train.ts", class_to_index), get_ds("data/basicmotions/processed/test.ts", class_to_index)

    loop = MultivariablePrototypeLearningTrainLoopReal(6,10,4,4,None).to(device)
    sequence = PretrainAndSaveLoop(loop).to(device)
    sequence.run(train_ds,test_ds,2000)

    # with torch.no_grad():
    #     visualize_moment = torch.utils.data.DataLoader(test_ds, len(test_ds))
    #     for train_sample  in visualize_moment:
    #         inp, out = train_sample[0].detach().to(device), train_sample[1].detach().to(device)
    #         num_vars = inp.shape[-1]
    #         for var in range(6):
    #             embeddings = sequence.pretrain_mod.pretrain.module_list[var](inp[:,:,var].unsqueeze(2).float())
    #             visualizer = UMAPLatent()
    #             visualizer.visualize(embeddings.to("cpu"),out.to("cpu"), 4)
    
    loop.mainloop(6400, train_ds, test_ds)

    with torch.no_grad():
        visualize_moment = torch.utils.data.DataLoader(test_ds, len(test_ds))
        for train_sample  in visualize_moment:
            inp, out = train_sample[0].detach().to(device), train_sample[1].detach().to(device)
            num_vars = inp.shape[-1]
            for var in range(6):
                embeddings = sequence.pretrain_mod.pretrain.module_list[var](inp[:,:,var].unsqueeze(2).float())
                embeddings = torch.concat([embeddings,loop.framework.signal_prototypes[var].protolayer.prototype_matrix], dim = 0)
                
                out = torch.concat([out, 4*torch.ones((loop.framework.signal_prototypes[var].protolayer.prototype_matrix.shape[0],)).to(device)],dim=0)
                visualizer = UMAPLatent()
                visualizer.visualize(embeddings.to("cpu"),out.to("cpu"), 4)
    
    loop2 = MultivariablePrototypeLearningTrainLoopSecond(loop.framework.signal_prototypes,6,24,4,4).to(device)
    loop2.mainloop(6400,train_ds,test_ds)