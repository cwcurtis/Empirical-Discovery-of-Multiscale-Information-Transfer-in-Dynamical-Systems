import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import knn_mi_comp as mi 
import os 
from copy import copy, deepcopy
from multiprocessing import Pool

import matplotlib
plt.rcParams.update({ "text.usetex": True, "font.family": "serif" })
font = {'size': 16}
matplotlib.rc('font', **font)

###################################################################################################################################   ###################################################################################################################################  
class Target:
    
    def __init__(self, target: npt.NDArray[np.float64], mxlag: int, kneighbor: int, alpha_m: float):
        self.target = target
        self.mxlag = mxlag
        self.alpha_m = alpha_m
        self.kneighbor = kneighbor
        self.model = []
        self.lags = list(np.arange(1,self.mxlag+1))
        self.te_vals = []
        
        
    def build_target_model(self) -> None:
        PASS = True
        while(PASS):
            mxcmi = 0.
            FLAG = False
            for lag in self.lags:            
                forward, backward, shift = self.forward_backward_target_models(lag)         
                if len(self.model)==0:
                    mtinf = mi.miknn(forward, backward, self.kneighbor)            
                else:
                    prior = self.current_target_model(shift)
                    mtinf = mi.cmiknn(forward, backward, prior, self.kneighbor)                                    
                if mtinf > mxcmi:
                    mxcmi = mtinf
                    bstlag = lag
                    FLAG = True      
            if FLAG: 
                PASS = self.test_and_add_target(bstlag, mxcmi)
            else:
                PASS = False
        return None


    def prune_target_model(self) -> None:
    
        PASS = True
        while(PASS and len(self.model)>1):
            mncmi = 1e6    
            for cnt, lag in enumerate(self.model):
                forward, backward, shift = self.forward_backward_target_models(lag)
                prior = self.cut_target_model(shift, cnt)            
                mtinf = mi.cmiknn(forward, backward, prior, self.kneighbor)
                if mtinf < mncmi:
                    mncmi = mtinf
                    wrstlag = lag
                    wrstcnt = cnt
            PASS = self.test_and_cut_target(wrstlag, wrstcnt, mncmi)
        return None
        
        
    def cut_target_model(self, shift: int, cut: int) -> npt.NDArray[np.float64]:
        cut_model = self.model[:cut] + self.model[cut+1:]            
        prior = np.zeros((self.target[shift:].size,len(cut_model)), dtype=np.float64)
        for mcnt, mlag in enumerate(cut_model):
            prior[:, mcnt] = self.target[shift-mlag: -mlag]
        return prior
            

    def forward_target_shift(self, lag: int) -> list:
        shift = lag
        if len(self.model) > 0:
            shift = max( [ lag, max(self.model) ] )        
        forward = self.target[shift:].reshape(-1,1)        
        return [forward, shift]
        

    def forward_backward_target_models(self, lag: int) -> list:
        forward, shift = self.forward_target_shift(lag)
        backward = self.target[shift-lag: -lag].reshape(-1,1)
        return [forward, backward, shift]
        

    def current_target_model(self, shift: int) -> npt.NDArray[np.float64]:        
        prior = np.zeros((self.target[shift:].size,len(self.model)), dtype=np.float64)
        for mcnt, mlag in enumerate(self.model):
            prior[:, mcnt] = self.target[shift-mlag: -mlag]
        return prior

        
    def test_and_add_target(self, bstlag: int, mxcmi: float) -> bool:    
        mxforward, mxbackward, shift = self.forward_backward_target_models(bstlag)
        PASS = True    
        if len(self.model) == 0:
            pval = mi.mi_shuffle_test(mxforward, mxbackward, self.kneighbor, self.alpha_m, mxcmi)
        else:                        
            mxprior = self.current_target_model(shift)
            pval = mi.cmi_shuffle_test(mxforward, mxbackward, mxprior, self.kneighbor, self.alpha_m, mxcmi, 'ADD')    
        if pval < self.alpha_m and len(self.lags)>0:
            self.model.append(bstlag)
            self.lags.remove(bstlag)
            self.te_vals.append(mxcmi)        
        else:
            PASS = False    
        return PASS
        

    def test_and_cut_target(self, mnlag: int, wrstcnt: int, mncmi: float) -> bool:
        PASS = True        
        mxforward, mxbackward, shift = self.forward_backward_target_models(mnlag)    
        mxprior = self.cut_target_model(shift, wrstcnt)    
        pval = mi.cmi_shuffle_test(mxforward, mxbackward, mxprior, self.kneighbor, self.alpha_m, mncmi, 'CUT')    
        if pval < self.alpha_m:
            self.model.remove(mnlag)        
            del self.te_vals[wrstcnt]
        else:
            PASS = False
        return PASS


###################################################################################################################################   ################################################################################################################################### 
class Source(Target): # A source only exists relative to a target or existing target/sources model.  
    

    def __init__(self, source: npt.NDArray[np.float64], target: Target, mxlag: int, kneighbor: int, alpha_m: float):
        super().__init__(source, mxlag, kneighbor, alpha_m)
        self.my_target = target
 
    
    def build_source(self) -> None:
        PASS = True
        while(PASS):
            mxcmi = 0.
            FLAG = False
            for lag in self.lags:
                forward, backward, prior = self.add_source(lag)                    
                mtinf = mi.cmiknn(forward, backward, prior, self.kneighbor)            
                if mtinf > mxcmi:
                    mxcmi = mtinf
                    bstlag = lag
                    FLAG = True
            if FLAG:                           
                PASS = self.test_and_add_source(bstlag, mxcmi)
            else:
                PASS = False
        return None

    
    def prune_source(self) -> None:        
        PASS = True    
        while(PASS and len(self.model)>0):
            mncmi = 1e6
            for cnt, lag in enumerate(self.model):
                forward, shift = self.forward_target_source_model(lag)            
                backward = self.target[shift-lag:-lag].reshape(-1,1)
                prior = self.cut_target_source_model(shift, cnt)            
                mtinf = mi.cmiknn(forward, backward, prior, knghbr=self.kneighbor)
                if mtinf < mncmi:
                    mncmi = mtinf
                    wrstlag = lag
                    wrstcnt = cnt
            PASS = self.test_and_cut_source(wrstlag, wrstcnt, mncmi)
        return None
    
    
    def current_target_source_model(self, shift: int) -> npt.NDArray[np.float64]:    
        target_prior = self.my_target.current_target_model(shift)    
        source_prior = self.current_target_model(shift)    
        return np.concatenate((target_prior, source_prior), axis=1)
        

    def cut_target_source_model(self, shift: int, cut: int):
        target_prior = self.my_target.current_target_model(shift)    
        source_prior = self.cut_target_model(shift, cut)    
        return np.concatenate((target_prior, source_prior), axis=1)
            

    def forward_target_source_model(self, lag: int) -> list:    
        shift = max( [ lag, max(self.my_target.model)] )
        if len(self.model) > 0:
            shift = max( [ shift, max(self.model) ] )        
        forward = self.my_target.target[shift:].reshape(-1,1)
        return [forward, shift]

    
    def test_and_cut_source(self, mnlag: int, wrstcnt: int, mncmi: float) -> bool:        
        PASS = True        
        mxforward, shift = self.forward_target_source_model(mnlag)    
        mxbackward = self.target[shift-mnlag:-mnlag].reshape(-1,1)
        mxprior = self.cut_target_source_model(shift, wrstcnt)                                   
        pval = mi.cmi_shuffle_test(mxforward, mxbackward, mxprior, self.kneighbor, self.alpha_m, mncmi, 'CUT')
        if pval < self.alpha_m:
            self.model.remove(mnlag)     
            del self.te_vals[wrstcnt]
        else:
            PASS = False
        return PASS        

    
    def test_and_add_source(self, bstlag: int, mxcmi: float) -> bool:        
        PASS = True            
        mxforward, mxbackward, mxprior = self.add_source(bstlag)
        pval = mi.cmi_shuffle_test(mxforward, mxbackward, mxprior, self.kneighbor, self.alpha_m, mxcmi, 'ADD')    
        if pval < self.alpha_m and len(self.lags)>0:
            self.model.append(bstlag)
            self.lags.remove(bstlag)
            self.te_vals.append(mxcmi)        
        else:
            PASS = False    
        return PASS    


    def add_source(self, lag: int) -> list:    
        forward, shift = self.forward_target_source_model(lag)                     
        backward = self.target[shift-lag : -lag].reshape(-1,1)                        
        if len(self.model) == 0:
            prior = self.my_target.current_target_model(shift)
        else:
            prior = self.current_target_source_model(shift)                    
        return [forward, backward, prior]        

        
###################################################################################################################################   ################################################################################################################################### 
class Entropy_Graph:
                   

    def __init__(self, target: Target, sources: npt.NDArray[np.float64], mxlag: int, kneighbor: int, alpha_m: float):
        self.mxlag = mxlag
        self.kneighbor = kneighbor
        self.alpha_m = alpha_m
        self.my_target = target
        self.sources = []
        self.num_sources = sources.shape[1]
        for jj in range(sources.shape[1]):
            self.sources.append( Source(sources[:, jj], target, mxlag, kneighbor, alpha_m) )
        self.chosen_sources = []    
        
    
    def build_sources(self) -> None:    
        # build candidate model
        for cnt in range(len(self.sources)):
            next_source = deepcopy(self.sources[cnt])
            if len(self.chosen_sources) == 0:
                next_source.build_source()
                next_source.prune_source()                           
            else:
                self.build_next_source(next_source)
                self.prune_next_source(next_source)      
            if len(next_source.model) > 0:
                self.sources[cnt].model = [lag for lag in next_source.model]
                self.sources[cnt].te_vals = [te_val for te_val in next_source.te_vals]
                self.chosen_sources.append(cnt)
            
        return None


    def build_next_source(self, next_source: Source) -> None:
        PASS = True
        while(PASS):
            mxcmi = 0.        
            FLAG = False
            for lag in next_source.lags:
                forward, backward, prior = self.add_next_source(next_source, lag)
                mtinf = mi.cmiknn(forward, backward, prior, self.kneighbor)            
                if mtinf > mxcmi:
                    mxcmi = mtinf
                    bstlag = lag 
                    mxforward = forward
                    mxbackward = backward
                    mxprior = prior
                    FLAG = True
            if FLAG:                           
                pval = mi.cmi_shuffle_test(mxforward, mxbackward, mxprior, self.kneighbor, self.alpha_m, mxcmi, 'ADD')        
                if pval < self.alpha_m and len(next_source.lags)>0:
                    next_source.model.append(bstlag)
                    next_source.lags.remove(bstlag)
                    next_source.te_vals.append(mxcmi)        
                else:
                    PASS = False
            else:
                PASS = False    
        return None
    

    def test_and_add_sources(self, next_source: Source, mxcmi: float, mxlag: int) -> bool:        
        PASS = True          
        mxforward, mxbackward, mxprior = self.add_next_source(next_source, mxlag)       
        pval = mi.cmi_shuffle_test(mxforward, mxbackward, mxprior, self.kneighbor, self.alpha_m, mxcmi, 'ADD')        
        if pval < self.alpha_m and len(next_source.lags)>0:
            next_source.model.append(mxlag)
            next_source.lags.remove(mxlag)
            next_source.te_vals.append(mxcmi)        
        else:
            PASS = False    
        return PASS
    
    
    def prune_next_source(self, next_source) -> None:        
        PASS = True           
        while(PASS and len(next_source.model)>0):
            mncmi = 1e6        
            for cnt, lag in enumerate(next_source.model):
                forward, shift = self.forward_target_sources_model(lag, next_source)                    
                backward = next_source.target[shift-lag:-lag].reshape(-1,1)
                prior = self.cut_target_sources_model(next_source, shift, cnt)                  
                mtinf = mi.cmiknn(forward, backward, prior, knghbr=self.kneighbor)
                if mtinf < mncmi:
                    mncmi = mtinf
                    wrstlag = lag
                    wrstcnt = cnt    
                    mnforward = forward
                    mnbackward = backward
                    mnprior = prior 
            pval = mi.cmi_shuffle_test(mnforward, mnbackward, mnprior, self.kneighbor, self.alpha_m, mncmi, 'CUT')
            if pval < self.alpha_m:
                next_source.model.remove(wrstlag)  
                del next_source.te_vals[wrstcnt]
            else:
                PASS = False
        else:
            PASS = False
        return None

        
    def test_and_cut_sources(self, next_source: Source, mnlag: int, wrstcnt: int, mncmi: float) -> bool:        
        PASS = True        
        mxforward, shift = self.forward_target_sources_model(mnlag, next_source)    
        mxprior = self.cut_target_sources_model(next_source, shift, wrstcnt)                  
        mxbackward = next_source.target[shift-mnlag:-mnlag].reshape(-1,1)
        pval = mi.cmi_shuffle_test(mxforward, mxbackward, mxprior, self.kneighbor, self.alpha_m, mncmi, 'CUT')
        if pval < self.alpha_m:
            next_source.model.remove(mnlag)  
            del next_source.te_vals[wrstcnt]
        else:
            PASS = False
        return PASS
   
    
    def add_next_source(self, next_source: Source, lag: int) -> list:        
        forward, shift = self.forward_target_sources_model(lag, next_source)                                    
        prior = self.advance_target_sources_model(next_source, shift, lag=0)
        backward = next_source.target[shift-lag:-lag].reshape(-1,1)
        return [forward, backward, prior]
    

    def model_maker(self, current_source: Source, local_model: list, shift: int):
        prior = np.zeros((current_source.target[shift:].size,len(local_model)), dtype=np.float64)        
        for mcnt, mlag in enumerate(local_model):
            prior[:, mcnt] = current_source.target[shift-mlag: -mlag]
        return prior    


    def current_sources_and_models(self):
        tmp_sources = []        
        tmp_sources_model = []
        for src in self.chosen_sources:
            tmp_sources.append(self.sources[src])
            tmp_sources_model.append(self.sources[src].model)
        return [tmp_sources, tmp_sources_model]

    
    def target_prior_choices_model_maker(self, shift):
        # Build prior chosen models 
        prior_choices, prior_choices_model = self.current_sources_and_models()
        target_prior = self.my_target.current_target_model(shift)    
        prior = copy(target_prior)
        for cnt, source in enumerate(prior_choices):
            local_prior = self.model_maker(source, prior_choices_model[cnt], shift)
            prior = np.concatenate((prior, local_prior), axis=1)
        return prior         
        

    def forward_target_sources_model(self, lag: int, next_source: Source):
        shift = max([lag, max(self.my_target.model)])
        if len(self.chosen_sources) > 0:
            for src in self.chosen_sources:
                if len(self.sources[src].model) > 0:
                    shift = max([shift, max(self.sources[src].model)])
        if len(next_source.model) > 0:
            shift = max([shift, max(next_source.model)])
        forward = self.my_target.target[shift:].reshape(-1, 1)
        return forward, shift
        
        
    def advance_target_sources_model(self, next_source: Source, shift:int, lag:int):
        # This never gets called unless we have made prior source choices
        # Build prior chosen models 
        tmp_sources, tmp_sources_model = self.current_sources_and_models()
        if len(next_source.model) > 0:
            tmp_sources.append(next_source)
            tmp_next_sources_model = copy(next_source.model)
            if lag > 0:
                tmp_next_sources_model.append(lag)
            tmp_sources_model.append(tmp_next_sources_model)
        else:
            if lag > 0:
                tmp_sources.append(next_source)
                tmp_sources_model.append([lag])
        prior = self.my_target.current_target_model(shift)    
        for cnt, source in enumerate(tmp_sources):
            local_prior = self.model_maker(source, tmp_sources_model[cnt], shift)
            prior = np.concatenate((prior, local_prior), axis=1)
        return prior 
    

    def current_target_sources_model(self, shift:int):
        # This never gets called unless we have made prior source choices
        # Build prior chosen models 
        tmp_sources, tmp_sources_model = self.current_sources_and_models()
        prior = self.my_target.current_target_model(shift)    
        for cnt, source in enumerate(tmp_sources):
            local_prior = self.model_maker(source, tmp_sources_model[cnt], shift)
            prior = np.concatenate((prior, local_prior), axis=1)
        return prior 


    def current_sources_model(self, shift:int):
        # Build prior chosen models 
        Flag = True
        if len(self.chosen_sources) > 0: 
            tmp_sources, tmp_sources_model = self.current_sources_and_models()
            for cnt, source in enumerate(tmp_sources):
                if cnt == 0:
                    backward = self.model_maker(source, tmp_sources_model[0], shift)    
                else:
                    local_backward = self.model_maker(source, tmp_sources_model[cnt], shift)
                    backward = np.concatenate((backward, local_backward), axis=1)
        else:
            print("No Selected Sources")
            Flag = False
            backward = []
        return backward, Flag 

    
    def cut_target_sources_model(self, next_source: Source, shift:int, cut:int):
        #This never gets called unless we have made prior source choices
        tmp_sources, tmp_sources_model = self.current_sources_and_models()
        cut_next_source_model = next_source.model[:cut] + next_source.model[cut+1:]
        tmp_sources_model.append(cut_next_source_model)                
        prior = self.my_target.current_target_model(shift)    
        for cnt, source in enumerate(tmp_sources):
            local_prior = self.model_maker(source, tmp_sources_model[cnt], shift)
            prior = np.concatenate((prior, local_prior), axis=1)
        return prior 

        
    # analytics for fully built source models 

    
    def get_forward(self):
        shift = max(self.my_target.model)
        if len(self.chosen_sources) > 0:
            for src in self.chosen_sources:
                if len(self.sources[src].model) > 0:
                    shift = max([shift, max(self.sources[src].model)])
        forward = self.my_target.target[shift:].reshape(-1, 1)
        return forward, shift


    def information_content(self):
        forward, shift = self.get_forward()
        target_prior = self.my_target.current_target_model(shift)    
        backward, Flag = self.current_sources_model(shift)
        if Flag:
            curr_cmi = mi.cmiknn(forward, backward, target_prior, self.kneighbor)
        else: 
            curr_cmi = 0.
        return curr_cmi


    def information_content_from_chosen_sources(self):        
        forward, shift = self.get_forward()
        it_source_by_source = []
        target_prior = self.my_target.current_target_model(shift)    
        
        if len(self.chosen_sources) > 0:
            for choice in self.chosen_sources:
                current_source = self.sources[choice]
                source_backward = self.model_maker(current_source, current_source.model, shift)
                it_source_by_source.append(mi.cmiknn(forward, source_backward, target_prior, self.kneighbor))
        return it_source_by_source    
        

#################################################################################################################################
#################################################################################################################################
class Model_Graph:
                   

    def __init__(self, data, max_lag: int, kneighbor: int, alpha_m: float):
        self.max_lag = max_lag
        self.kneighbor = kneighbor
        self.alpha_m = alpha_m
        self.data = data
        self.num_scales = data.shape[0]
        self.target_models = []
        self.source_models = []
        self.source_te_vals = []
        self.chosen_sources_original_indices = []
        self.chosen_sources_for_target = []
        self.te_mat = np.zeros((data.shape[0], data.shape[0]))
        self.it_mat = np.zeros((data.shape[0], data.shape[0]))
        self.source_info_mat = np.zeros((data.shape[0], data.shape[0]))

    def et_graph_model_builder(self):
    
        for jj in range(self.num_scales):
            
            # Build target
            current_target = Target(self.data[jj, :], self.max_lag, self.kneighbor, self.alpha_m)
            current_target.build_target_model()
            current_target.prune_target_model()    
            self.target_models.append(current_target)    
            self.sources_from_built_target(current_target, jj)
            
        return None

    
    def et_graph_model_builder_fixed_target_model(self):
    
        for jj in range(self.num_scales):
            
            # Build target
            current_target = Target(self.data[jj, :], 1, self.kneighbor, self.alpha_m)
            current_target.model=[self.max_lag]
            self.target_models.append(current_target)    
            self.sources_from_built_target(current_target, jj)
            
        return None
    

    def sources_from_built_target(self, current_target, jj):
        # Begin building out sources 
        source_inds = list(np.arange(0,jj)) + list(np.arange(jj+1,self.num_scales))
        sources = self.data[source_inds, :].T        
        one_source_cmi = np.zeros(len(source_inds), dtype=np.float64)
        shift = max(current_target.model)
        forward = current_target.target[shift:].reshape(-1,1)
        if len(current_target.model) > 0:
            for cnt, val in enumerate(current_target.model):
                if cnt == 0:
                    prior = current_target.target[shift-val:-val].reshape(-1,1)
                else:
                    local_prior = current_target.target[shift-val:-val].reshape(-1,1)
                    prior = np.concatenate((prior, local_prior),1)
        else:
            prior = current_target.target[:-shift].reshape(-1,1)
            
        # Rank sources from most to least informative 
        for ll, source_ind in enumerate(source_inds):
            backward = sources[:-shift, ll].reshape(-1, 1)
            one_source_cmi[ll] = mi.cmiknn(forward, backward, prior, self.kneighbor)
            self.source_info_mat[jj, source_ind] = max([one_source_cmi[ll], 0.])
                
        sort_inds = list(np.argsort(one_source_cmi)[::-1])
        source_inds = [source_inds[ind] for ind in sort_inds]
        sort_sources = self.data[source_inds, :].T
    
        # Build full graph following ranked source order 
        best_entropy_graph = Entropy_Graph(current_target, sort_sources, self.max_lag, self.kneighbor, self.alpha_m)
        best_entropy_graph.build_sources()
        final_info = best_entropy_graph.information_content()        
            
        # Now assign final models.  
        if len(best_entropy_graph.chosen_sources) > 0:
            self.source_models.append(best_entropy_graph)
            original_indices = [source_inds[choice] for choice in best_entropy_graph.chosen_sources]
            self.chosen_sources_original_indices.append(original_indices)
            self.chosen_sources_for_target.append(jj)
            source_te_vals = [best_entropy_graph.sources[choice].te_vals for choice in best_entropy_graph.chosen_sources ]
            max_te_vals = [max(te_val) for te_val in source_te_vals if len(te_val)>0] 
            self.source_te_vals.append(source_te_vals)
            self.te_mat[jj, original_indices] = max_te_vals
            self.it_mat[jj, original_indices] = best_entropy_graph.information_content_from_chosen_sources()
            
        print(f"For target {jj}")
        print(f"Target model: {current_target.model}")
        print(f"Chosen sources: {self.chosen_sources_original_indices}")
        print(f"Final information content of model: {final_info}")
        print()
        return None        

    
    def mat_visualizer(self, matrix, variable_label, log, cmap):

        labels = [None]*self.num_scales
        for jj in range(self.num_scales):
            labels[jj] = r"$" + variable_label + "_{" + str(jj) + "}$"
        tick_position = np.arange(self.num_scales)
        fig, ax = plt.subplots()       
        if log:
            image = ax.imshow(np.ma.log10(matrix), cmap=cmap)
        else:
            image = ax.imshow(matrix, cmap=cmap)
        ax.set_xticks(tick_position, labels=labels)
        ax.set_yticks(tick_position, labels=labels)
        plt.colorbar(image)        
        
        return None 

    
###################################################################################################################################   ################################################################################################################################### 


def model_build_and_mi_comp(current_entropy_graph):
    current_entropy_graph.build_sources()
    return current_entropy_graph


def et_graph_model_builder_parallel(data, max_lag, kneighbor, alpha_m):
    num_scales = data.shape[0]
    target_models = []
    source_models = []
    source_te_vals = []
    chosen_sources_original_indices = []
    num_cpus = os.cpu_count()
    print(f"Number of Processors: {num_cpus}")
        
    for jj in range(num_scales):
        # Build Target
        current_target = Target(data[jj, :], max_lag, kneighbor, alpha_m)
        current_target.build_target_model()
        #current_target.prune_target_model()    
        target_models.append(current_target)

        source_inds = list(np.arange(0,jj)) + list(np.arange(jj+1,num_scales))
        orig_source_inds = copy(source_inds)    
        entropy_graphs_list = [None]*num_cpus
        
        for kk in range(num_cpus): 
            # Shuffle input sources to remove any dependence on ordering 
            rng = np.random.default_rng()
            rng.shuffle(source_inds)
            sources = data[source_inds, :].T
            # Build initial model         
            entropy_graphs_list[kk] = Entropy_Graph(current_target, sources, max_lag, kneighbor, alpha_m)
        
        print("Building Models")
        with Pool() as pool:
            results = pool.map(model_build_and_mi_comp, entropy_graphs_list)

        max_info = 0.
        for ll in range(num_cpus):
            current_model = results[ll]
            current_info = current_model.information_content()
            if current_info > max_info:
                max_info = current_info
                best_entropy_graph = current_model
            
        print(f"Most Informative Model: {max_info}")
        
        # Now assign final models.  
        source_models.append(best_entropy_graph)
        chosen_sources_original_indices.append([source_inds[chc] for chc in best_entropy_graph.chosen_sources])
        source_te_vals.append([best_entropy_graph.sources[cnt].te_vals for cnt in range(len(source_inds)) ])
        
        print(f"For target {jj}")
        print(current_target.model)
        print(current_target.te_vals)
        print(chosen_sources_original_indices)

    Admat = np.zeros((num_scales,num_scales))
    for ll in range(num_scales):
        Admat[ll, chosen_sources_original_indices[ll][:]] = 1

    return target_models, source_models, source_te_vals, chosen_sources_original_indices, Admat