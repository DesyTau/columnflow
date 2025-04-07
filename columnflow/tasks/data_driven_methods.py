
"""
Task to produce and merge histograms.
"""

from __future__ import annotations

import luigi
import law

from columnflow.tasks.framework.base import Requirements, AnalysisTask, DatasetTask, wrapper_factory
from columnflow.tasks.framework.mixins import (
    CalibratorsMixin, SelectorStepsMixin, ProducersMixin, MLModelsMixin, VariablesMixin,
    ShiftSourcesMixin, WeightProducerMixin, ChunkedIOMixin, DatasetsProcessesMixin, CategoriesMixin
)
from columnflow.tasks.framework.plotting import ProcessPlotSettingMixin

from columnflow.tasks.framework.remote import RemoteWorkflow
from columnflow.tasks.framework.parameters import last_edge_inclusive_inst
from columnflow.tasks.reduction import ReducedEventsUser
from columnflow.tasks.production import ProduceColumns
from columnflow.tasks.ml import MLEvaluation
from columnflow.util import dev_sandbox, DotDict


class PrepareFakeFactorHistograms(
    CategoriesMixin,
    WeightProducerMixin,
    MLModelsMixin,
    ProducersMixin,
    ReducedEventsUser,
    ChunkedIOMixin,
    law.LocalWorkflow,
    RemoteWorkflow,
):
    last_edge_inclusive = last_edge_inclusive_inst

    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    # upstream requirements
    reqs = Requirements(
        ReducedEventsUser.reqs,
        RemoteWorkflow.reqs,
        ProduceColumns=ProduceColumns,
    )

    # strategy for handling missing source columns when adding aliases on event chunks
    missing_column_alias_strategy = "original"

    # names of columns that contain category ids
    # (might become a parameter at some point)
    category_id_columns = {"category_ids"}

    # register sandbox and shifts found in the chosen weight producer to this task
    register_weight_producer_sandbox = True
    register_weight_producer_shifts = True

    @law.util.classproperty
    def mandatory_columns(cls) -> set[str]:
        return set(cls.category_id_columns) | {"process_id"}

    # def create_branch_map(self):
    #     # create a dummy branch map so that this task could be submitted as a job
    #     return {0: None}
    
    def workflow_requires(self):
        reqs = super().workflow_requires()

        # require the full merge forest
        reqs["events"] = self.reqs.ProvideReducedEvents.req(self)

        if not self.pilot:
            if self.producer_insts:
                reqs["producers"] = [
                    self.reqs.ProduceColumns.req(self, producer=producer_inst.cls_name)
                    for producer_inst in self.producer_insts
                    if producer_inst.produced_columns
                ]

            # add weight_producer dependent requirements
            reqs["weight_producer"] = law.util.make_unique(law.util.flatten(self.weight_producer_inst.run_requires()))

        return reqs

    def requires(self):
        reqs = {"events": self.reqs.ProvideReducedEvents.req(self)}

        if self.producer_insts:
            reqs["producers"] = [
                self.reqs.ProduceColumns.req(self, producer=producer_inst.cls_name)
                for producer_inst in self.producer_insts
                if producer_inst.produced_columns
            ]

        # add weight_producer dependent requirements
        reqs["weight_producer"] = law.util.make_unique(law.util.flatten(self.weight_producer_inst.run_requires()))

        return reqs

    workflow_condition = ReducedEventsUser.workflow_condition.copy()

    @workflow_condition.output
    def output(self):
        return  {"hists": self.target(f"ff_hist_{self.branch}.pickle")}
    @law.decorator.notify
    @law.decorator.log
    @law.decorator.localize(input=True, output=False)
    @law.decorator.safe_output
    def run(self):
        import hist
        import numpy as np
        import awkward as ak
        from columnflow.columnar_util import (
            Route, update_ak_array, add_ak_aliases, has_ak_column, attach_coffea_behavior, EMPTY_FLOAT
        )
        from columnflow.hist_util import fill_hist
        # prepare inputs
        inputs = self.input()

        # declare output: dict of histograms
        histograms = {}

        # run the weight_producer setup
        producer_reqs = self.weight_producer_inst.run_requires()
        reader_targets = self.weight_producer_inst.run_setup(producer_reqs, luigi.task.getpaths(producer_reqs))

        # create a temp dir for saving intermediate files
        tmp_dir = law.LocalDirectoryTarget(is_tmp=True)
        tmp_dir.touch()

        # get shift dependent aliases
        aliases = self.local_shift_inst.x("column_aliases", {})
        ff_variables = [var.var_route for var in self.config_inst.x.fake_factor_method.axes.values()]
        # define columns that need to be read
        
        read_columns = {Route("process_id")}
        read_columns |= set(map(Route, self.category_id_columns))
        read_columns |= set(self.weight_producer_inst.used_columns)
        read_columns |= set(map(Route, aliases.values()))
        read_columns |= set(map(Route, ff_variables))
        # empty float array to use when input files have no entries
        empty_f32 = ak.Array(np.array([], dtype=np.float32))

        # iterate over chunks of events and diffs
        file_targets = [inputs["events"]["events"]]
        if self.producer_insts:
            file_targets.extend([inp["columns"] for inp in inputs["producers"]])
            
        # prepare inputs for localization
        with law.localize_file_targets(
            [*file_targets, *reader_targets.values()],
            mode="r",
        ) as inps:
            
            for (events, *columns), pos in self.iter_chunked_io(
                [inp.abspath for inp in inps],
                source_type=len(file_targets) * ["awkward_parquet"] + [None] * len(reader_targets),
                read_columns=(len(file_targets) + len(reader_targets)) * [read_columns],
                chunk_size=self.weight_producer_inst.get_min_chunk_size(),
            ):
                # optional check for overlapping inputs
                if self.check_overlapping_inputs:
                    self.raise_if_overlapping([events] + list(columns))
                # add additional columns
                events = update_ak_array(events, *columns)
                # add aliases
                events = add_ak_aliases(
                    events,
                    aliases,
                    remove_src=True,
                    missing_strategy=self.missing_column_alias_strategy,
                )

                # attach coffea behavior aiding functional variable expressions
                events = attach_coffea_behavior(events)
                # build the full event weight
                if hasattr(self.weight_producer_inst, "skip_func") and not self.weight_producer_inst.skip_func():
                    events, weight = self.weight_producer_inst(events)
                else:
                    weight = ak.Array(np.ones(len(events), dtype=np.float32))
                # define and fill histograms, taking into account multiple axes
                category_ids = ak.concatenate(
                        [Route(c).apply(events) for c in self.category_id_columns],
                        axis=-1,)
                sr_names = self.categories
                for sr_name in sr_names:
                    the_sr = self.config_inst.get_category(sr_name)
                    regions = [sr_name]
                    if the_sr.aux:
                        for the_key in the_sr.aux.keys():
                            if (the_key == 'abcd_regs') or (the_key == 'ff_regs'):
                                regions += list(the_sr.aux[the_key].values())
                    else:
                        raise KeyError(f"Application and determination regions are not found for {the_sr}. \n Check aux field of the category map!") 
    
                    for region in regions: 
                        #by accessing the list of categories we check if the category with this name exists
                        cat = self.config_inst.get_category(region)
                        
                        # get variable instances
                        mask = ak.any(category_ids == cat.id, axis = 1)
                        masked_events = events[mask]
                        masked_weight = weight[mask]
                        
                        h = (hist.Hist.new.IntCat([], name="process", growth=True))
                        for (var_name, var_axis) in self.config_inst.x.fake_factor_method.axes.items(): 
                            h = eval(f'h.{var_axis.ax_str}') 
                        
                        h = h.Weight()
                        # broadcast arrays so that each event can be filled for all its categories
                        
                        fill_data = {
                            "process": masked_events.process_id,
                            "weight"  : masked_weight,
                        }
                        for (var_name, var_axis) in self.config_inst.x.fake_factor_method.axes.items(): 
                            route = Route(var_axis.var_route)
                            if len(masked_events) == 0 and not has_ak_column(masked_events, route):
                                values = empty_f32
                            else:
                                values = route.apply(masked_events)
                                if values.ndim != 1: values = ak.firsts(values,axis=1)
                                values = ak.fill_none(values, EMPTY_FLOAT)
                                
                                if var_name == 'n_jets': values = ak.where (values > 2, 
                                                                            2 * ak.ones_like(values),
                                                                            values) 
                                
                                if 'Int' in var_axis.ax_str: values = ak.values_astype(values, np.int64)
                            fill_data[var_name] = values
                        # fill it
                        fill_hist(
                            h,
                            fill_data,
                        )
                        if cat.name not in histograms.keys():
                            histograms[cat.name] = h
                        else:
                            histograms[cat.name] +=h
                        
        # merge output files
        self.output()["hists"].dump(histograms, formatter="pickle")
    
   


# overwrite class defaults
check_overlap_tasks = law.config.get_expanded("analysis", "check_overlapping_inputs", [], split_csv=True)
PrepareFakeFactorHistograms.check_overlapping_inputs = ChunkedIOMixin.check_overlapping_inputs.copy(
    default=PrepareFakeFactorHistograms.task_family in check_overlap_tasks,
    add_default_to_description=True,
)


PrepareFakeFactorHistogramsWrapper = wrapper_factory(
    base_cls=AnalysisTask,
    require_cls=PrepareFakeFactorHistograms,
    enable=["configs", "skip_configs", "datasets", "skip_datasets", "shifts", "skip_shifts"],
)


class MergeFakeFactorHistograms(
    #VariablesMixin,
    #WeightProducerMixin,
    #MLModelsMixin,
    #ProducersMixin,
    #SelectorStepsMixin,
    #CalibratorsMixin,
    DatasetTask,
    law.LocalWorkflow,
    RemoteWorkflow,
):
    only_missing = luigi.BoolParameter(
        default=False,
        description="when True, identify missing variables first and only require histograms of "
        "missing ones; default: False",
    )
    remove_previous = luigi.BoolParameter(
        default=False,
        significant=False,
        description="when True, remove particlar input histograms after merging; default: False",
    )

    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    # upstream requirements
    reqs = Requirements(
        RemoteWorkflow.reqs,
        PrepareFakeFactorHistograms=PrepareFakeFactorHistograms,
    )

    @classmethod
    def req_params(cls, inst: AnalysisTask, **kwargs) -> dict:
        _prefer_cli = law.util.make_set(kwargs.get("_prefer_cli", [])) | {"variables"}
        kwargs["_prefer_cli"] = _prefer_cli
        return super().req_params(inst, **kwargs)

    def create_branch_map(self):
        # create a dummy branch map so that this task could be submitted as a job
        return {0: None}

    # def _get_variables(self):
    #     if self.is_workflow():
    #         return self.as_branch()._get_variables()

    #     variables = self.variables

    #     # optional dynamic behavior: determine not yet created variables and require only those
    #     if self.only_missing:
    #         missing = self.output().count(existing=False, keys=True)[1]
    #         variables = sorted(missing, key=variables.index)

    #     return variables

    def workflow_requires(self):
        reqs = super().workflow_requires()

        if not self.pilot:
            #variables = self._get_variables()
            #if variables:
            reqs["hists"] = self.reqs.PrepareFakeFactorHistograms.req_different_branching(
                    self,
                    branch=-1,
                    #variables=tuple(variables),
            )

        return reqs

    def requires(self):
        #variables = self._get_variables()
        #if not variables:
        #    return []

        return self.reqs.PrepareFakeFactorHistograms.req_different_branching(
            self,
            branch=-1,
            #variables=tuple(variables),
            workflow="local",
        )

    def output(self):
        return {"hists": self.target(f"merged_ff_hist.pickle")}

    @law.decorator.notify
    @law.decorator.log
    def run(self):
        # preare inputs and outputs
        inputs = self.input()["collection"]
        outputs = self.output()

        # load input histograms
        hists = [
            inp["hists"].load(formatter="pickle")
            for inp in self.iter_progress(inputs.targets.values(), len(inputs), reach=(0, 50))
        ]
        cats = list(hists[0].keys())
        get_hists = lambda hists, cat : [h[cat] for h in hists]
        # create a separate file per output variable
        merged_hists = {}
        self.publish_message(f"merging {len(hists)} histograms for {self.dataset}")
        for the_cat in cats:
            h = get_hists(hists, the_cat)
            merged_hists[the_cat] = sum(h[1:], h[0].copy())
        outputs["hists"].dump(merged_hists, formatter="pickle")
        # optionally remove inputs
        if self.remove_previous:
            inputs.remove()

MergeFakeFactorHistogramsWrapper = wrapper_factory(
    base_cls=AnalysisTask,
    require_cls=MergeFakeFactorHistograms,
    enable=["configs", "skip_configs", "datasets", "skip_datasets", "shifts", "skip_shifts"],
)

class ComputeFakeFactors(
    DatasetsProcessesMixin,
    CategoriesMixin,
    WeightProducerMixin,
    ProducersMixin,
):
    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    only_missing = luigi.BoolParameter(
        default=False,
        description="when True, identify missing variables first and only require histograms of "
        "missing ones; default: False",
    )
    remove_previous = luigi.BoolParameter(
        default=False,
        significant=False,
        description="when True, remove particlar input histograms after merging; default: False",
    )
    
    # upstream requirements
    reqs = Requirements(
        RemoteWorkflow.reqs,
        MergeFakeFactorHistograms=MergeFakeFactorHistograms,
    )
    
    def store_parts(self):
        parts = super().store_parts()
        parts.insert_before("version", "datasets", f"datasets_{self.datasets_repr}")
        return parts
    
    @classmethod
    def req_params(cls, inst: AnalysisTask, **kwargs) -> dict:
        _prefer_cli = law.util.make_set(kwargs.get("_prefer_cli", [])) | {"variables"}
        kwargs["_prefer_cli"] = _prefer_cli
        return super().req_params(inst, **kwargs)
    
    def create_branch_map(self):
        # create a dummy branch map so that this task could be submitted as a job
        return {0: None}

        return reqs
    def requires(self):
        return {
            d: self.reqs.MergeFakeFactorHistograms.req_different_branching(
                self,
                branch=-1,
                dataset=d,
                workflow="local",
            )
            for d in self.datasets
        }
    
    def output(self):
        year = self.config_inst.campaign.aux['year']
        tag = self.config_inst.campaign.aux['tag']
        channel = self.config_inst.channels.get_first().name
        return {"ff_json": self.target('_'.join(('fake_factors',
                                                 channel,
                                                 str(year),
                                                 tag)) + '.json'),
                "plots": {'_'.join((ff_type,
                                    syst,
                                    f'n_jets_{str(nj)}')): self.target(f"fake_factor_{ff_type}_{syst}_njets_{str(nj)}.png")
                          for syst in ['nominal', 'up', 'down']
                          for ff_type in ['qcd','wj']
                          for nj in [0,1,2]},
                "plots1d": {'_'.join((ff_type,
                                      str(dm),
                                      str(nj))): self.target(f"fake_factor_{ff_type}_PNet_dm_{str(dm)}_njets_{str(nj)}.png")
                          for ff_type in ['qcd','wj']
                          for dm in [0,1,2,10,11]
                          for nj in [0,1,2]},
                "fitres": self.target('_'.join(('fitres',
                                                 channel,
                                                 str(year),
                                                 tag)) + '.json'),
                }

    @law.decorator.log
    def run(self):
        import hist
        import numpy as np
        from scipy.optimize import curve_fit
        from scipy.special import erf
        import matplotlib.pyplot as plt
        import correctionlib.schemav2 as cs
        from numpy import exp
        plt.figure(dpi=200)
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "monospace",
            "font.monospace": 'Computer Modern Typewriter'
        })
        # preare inputs and outputs
        inputs = self.input()
        outputs = self.output()
        
        hists_by_dataset = []
        merged_hists = {}
        for (dataset_name, dataset) in inputs.items():
            files = dataset['collection'][0]
            
            # load input histograms per dataset
            input_chunked_hists = []
            input_chunked_hists = [f.load(formatter='pickle') for f in files.values()]
            
            for hists in input_chunked_hists:
                for the_cat, the_hist in hists.items():
                    if the_cat not in merged_hists.keys():
                        merged_hists[the_cat] = []
                        merged_hists[the_cat].append(the_hist)
                    else:
                        merged_hists[the_cat].append(the_hist)
        
        #merge histograms
        mc_hists = {}
        data_hists = {}
        #devide between data and mc
        for the_cat, h_list in merged_hists.items():
            for the_hist in h_list:
                for proc_name in self.config_inst.processes.names():
                    proc = self.config_inst.processes.get(proc_name)
                    if proc.id in the_hist.axes["process"]: 
                        h = the_hist.copy()
                        h = h[{"process": hist.loc(proc.id)}]
                        if proc.is_mc and not proc.has_tag("signal"):
                            if the_cat in mc_hists: mc_hists[the_cat] += h
                            else: mc_hists[the_cat] = h
                        if proc.is_data:
                            if the_cat in data_hists: data_hists[the_cat] += h
                            else: data_hists[the_cat] = h
        
        def eval_formula(formula_str, popt,make_rounding=False):
                for i,p in enumerate(popt):
                    if make_rounding:
                        formula_str = formula_str.replace(f'p{i}', '{:.3e}'.format(p))
                    else:
                        formula_str = formula_str.replace(f'p{i}',str(p))
                return formula_str
        
        #Function that performs the calculation of t
        def get_ff_corr(self, h_data, h_mc, dr_num, dr_den, name='ff_hist', label='ff_hist'):
            
            def get_single_cat(self, h, reg_name): 
                cat_name = self.config_inst.get_category(self.categories[0]).aux['ff_regs'][reg_name]
                return h[cat_name]
            data_num = get_single_cat(self, h_data, dr_num)
            data_den = get_single_cat(self, h_data, dr_den)
            mc_num = get_single_cat(self, h_mc, dr_num)
            mc_den = get_single_cat(self, h_mc, dr_den)
            print(name)
            for nj in [0,1,2]:
                for dm in [0,1,2,10,11]:
                    print(f'DM {dm} Nj {nj}')
                    print(f"data_num: {data_num[{'tau_dm_pnet': hist.loc(dm), 'n_jets': hist.loc(nj)}].values()}")
                    print(f"data_den: {data_den[{'tau_dm_pnet': hist.loc(dm), 'n_jets': hist.loc(nj)}].values()}")
                    print(f"mc_num: {mc_num[{'tau_dm_pnet': hist.loc(dm), 'n_jets': hist.loc(nj)}].values()}")
                    print(f"mc_den: {mc_den[{'tau_dm_pnet': hist.loc(dm), 'n_jets': hist.loc(nj)}].values()}")
            num = data_num.values() - mc_num.values()

            den = data_den.values() - mc_den.values()
            ff_val = np.where((num > 0) & (den > 0),
                               num / np.maximum(den, 1),
                               -1)
            def rel_err(x):
                return x.variances()/np.maximum(x.values()**2, 1)
            
            ff_err = ff_val * ((data_num.variances() + mc_num.variances())**0.5 / np.abs(num) + (data_den.variances() + mc_den.variances())**0.5 / np.abs(den))
            
            ff_err[ff_val < 0] = 1
            h = hist.Hist.new
            for (var_name, var_axis) in self.config_inst.x.fake_factor_method.axes.items(): 
                h = eval(f'h.{var_axis.ax_str}') 
            axes = list(h.axes[1:])
            h = h.StrCategory(['nominal', 'up', 'down'], name='syst', label='Statistical uncertainty of the fake factor')
            ff_raw = h.Weight()
            ff_raw.view().value[...,0] = ff_val
            ff_raw.view().variance[...,0] = ff_err**2
            ff_raw.name = name + '_raw'
            ff_raw.label = label + '_raw'
            
            def get_fitf(dm):
                if dm==0:
                    formula_str = 'p0+p1*x+p2*x*x'
                    def fitf(x,p0,p1,p2): 
                        return eval(formula_str)
                else:
                    formula_str = 'p0+p1*exp(-p2*x)'
                    def fitf(x,p0,p1,p2): 
                        from numpy import exp
                        return eval(formula_str)
                return fitf, formula_str
         
            def get_jac(dm):
                if dm==0:
                    def jac(x,p): 
                        from numpy import array
                        return array([ 1., x, x**2])
                else:
                    def jac(x,p):
                        from numpy import array,exp,outer
                        ders=array([ 1.,
                                    exp(-p[2]*x),
                                    -1*p[1]*x*exp(-p[2]*x)])
                        return ders
                return jac 
            
            ff_fitted = ff_raw.copy().reset()
            ff_fitted.name = name
            ff_fitted.label = label
            
            fitres = {}
            dm_axis = ff_raw.axes['tau_dm_pnet']
            n_jets_axis = ff_raw.axes['n_jets']
            
            for nj in n_jets_axis:
                if nj not in fitres.keys(): fitres[nj] = {}
                for dm in dm_axis:
                    if dm not in fitres[nj].keys(): fitres[nj][dm] = {}
                    
                 
                        
                    
                    h1d = ff_raw[{'tau_dm_pnet': hist.loc(dm),
                                   'n_jets': hist.loc(nj),
                                    'syst': hist.loc('nominal')}]
                    mask = h1d.values() > 0
                    x = h1d.axes[0].centers
                    if np.sum(mask) < 2:
                        y = np.zeros_like(x)
                        y_err = np.ones_like(x)
                        x_masked = x
                    else:
                        y = h1d.values()[mask]
                        y_err = (h1d.variances()[mask])**0.5
                        x_masked = x[mask]
                    
                    fitf, formula_str = get_fitf(dm)
                    if dm==0:
                        the_bounds = ([-10,-5,-1],[10,5,1])
                    else:
                        the_bounds = ([-0.5, -1, 0],[0.5,1,0.1])
                    popt, pcov, infodict, mesg, ier = curve_fit(fitf,
                                           x_masked,
                                           y,
                                           sigma=y_err,
                                           bounds=the_bounds,
                                           absolute_sigma=True,
                                           full_output=True
                                        )
                    fitres[nj][dm]['chi2']      = sum((infodict['fvec'])**2)
                    fitres[nj][dm]['ndf']       = len(y) - len(popt)
                    fitres[nj][dm]['popt']      = popt 
                    fitres[nj][dm]['pcov']      = pcov
                    fitres[nj][dm]['x_max']     = np.max(x_masked)
                   
                    fitres[nj][dm]['jac']       = get_jac(dm)
                    fitres[nj][dm]['name']      = name
                    fitres[nj][dm]['fitf']      = fitf
                    fitres[nj][dm]['fitf_str']  = formula_str
                    
                    for c, shift_name in enumerate(['down', 'nominal', 'up']): # if down then c=-1, if up c=+1, nominal => c=0
                        ff_fitted.view().value[:,
                                            ff_fitted.axes[1].index(dm),
                                            ff_fitted.axes[2].index(nj),
                                            ff_fitted.axes[3].index(shift_name)] = fitf(x, *popt + (c-1) * np.sqrt(np.diag(pcov)))
                        
            return ff_raw, ff_fitted, fitres
        
        wj_raw, wj_fitted, wj_fitres = get_ff_corr(self,
                              data_hists,
                              mc_hists,
                              dr_num = 'dr_num_wj',
                              dr_den = 'dr_den_wj',
                              name='ff_wjets',
                              label='Fake factor W+jets')
        
        qcd_raw, qcd_fitted, qcd_fitres = get_ff_corr(self,
                              data_hists,
                              mc_hists,
                              dr_num = 'dr_num_qcd',
                              dr_den = 'dr_den_qcd',
                              name='ff_qcd',
                              label='Fake factor QCD')
        
        
        corr_list = []            
        for fitres_per_proc in [wj_fitres, qcd_fitres]:
            nj_categories = []
            for nj, fitres_per_nj in fitres_per_proc.items():
                single_nj = []
                for dm, fitres in fitres_per_nj.items():
                    x_max = fitres['x_max']
                    fitf = fitres['fitf']
                    popt = fitres['popt']
                    fitf_str = eval_formula(fitres['fitf_str'], popt)
                    fx_max = np.maximum(fitf(x_max,*popt),0)
                    single_nj.append(cs.CategoryItem(
                        key=dm,
                        value=cs.Formula(
                            nodetype="formula",
                            variables=["tau_pt"],
                            parser="TFormula",
                            expression=f'({fitf_str})*((x-{x_max})<0) + ({fx_max})*((x-{x_max})>=0)',
                        )))
                nj_categories.append(cs.CategoryItem(
                        key=nj,
                        value=cs.Category(
                            nodetype="category",
                            input="tau_dm_pnet",
                            content=single_nj,
                            )))
            corr_list.append(cs.Correction(
                name=fitres_per_proc[0][0]['name'],
                description=f"fake factor correcton for {fitres_per_proc[0][0]['name'].split('_')[1]}",
                version=2,
                inputs=[
                    cs.Variable(name="tau_pt", type="real",description="pt of tau"),
                    cs.Variable(name="tau_dm_pnet", type="int", description="PNet decay mode of tau"),
                    cs.Variable(name="n_jets", type="int", description="Number of jets with pt > 20 GeV and eta < 4.7"),
                ],
                output=cs.Variable(name="weight", type="real", description="Multiplicative event weight"),
                data=cs.Category(
                    nodetype="category",
                    input="n_jets",
                    content=nj_categories,
                )
            ))
        cset = cs.CorrectionSet(
        schema_version=2,
        description="Fake factors",
        corrections=corr_list
        )
        self.output()['ff_json'].dump(cset.json(exclude_unset=True), formatter="json")
        
        chi2_string = 'type nj dm chi2 ndf,'
        for fitres_per_proc in [wj_fitres, qcd_fitres]:
            for dm, fitres_per_dm in fitres_per_proc.items():
                for nj, fitres in fitres_per_dm.items():
                    chi2_string += ' '.join((fitres['name'],
                                             str(nj),
                                             str(dm),
                                             str(fitres['chi2']),
                                             str(fitres['ndf'])))
                    chi2_string += ','
        self.output()['fitres'].dump(chi2_string, formatter="json")
        
        #Plot fake factors:
        for h_name in ['wj', 'qcd']:
            h_raw       = eval(f'{h_name}_raw')
            h_fitted    = eval(f'{h_name}_fitted')
            fitres_dict = eval(f'{h_name}_fitres')
            dm_axis     = h_raw.axes['tau_dm_pnet']
            nj_axis     = h_raw.axes['n_jets']
            for nj in nj_axis:
                print(f"Plotting 2d map for n jets = {nj}")
                fig, ax = plt.subplots(figsize=(12, 8))
                
                single2d_h = h_raw[{'n_jets': hist.loc(nj),
                       'syst': hist.loc('nominal')}]
                pcm = ax.pcolormesh(*np.meshgrid(*single2d_h.axes.edges), single2d_h.view().value.T, cmap="viridis", vmin=0, vmax=0.5)
                ax.set_yticks(dm_axis.centers, labels=list(map(dm_axis.bin, range(dm_axis.size))))
                plt.colorbar(pcm, ax=ax)
                plt.xlabel(single2d_h.axes.label[0])
                plt.ylabel(single2d_h.axes.label[1])
                plt.title(single2d_h.label)

                self.output()['plots']['_'.join((h_name,'nominal',f'n_jets_{str(nj)}'))].dump(fig, formatter="mpl")
                for dm in dm_axis:
                    print(f"Plotting 1d plot for n jets = {nj}, dm = {dm}")
                    h1d = h_raw[{'tau_dm_pnet': hist.loc(dm),
                                 'n_jets': hist.loc(nj),
                                    'syst': hist.loc('nominal')}]
                    hfit = h_fitted[{'tau_dm_pnet': hist.loc(dm),
                                     'n_jets': hist.loc(nj),}]
                    fig, ax = plt.subplots(figsize=(8, 6))
                    mask = h1d.counts() > 0
                    if np.sum(mask) > 0: 
                        x = h1d.axes[0].centers[mask]
                        y = h1d.counts()[mask]
                        xerr = (np.diff(h1d.axes[0]).flatten()/2.)[mask],
                        yerr = np.sqrt(h1d.variances()).flatten()[mask],
                    else:
                        x = h1d.axes[0].centers
                        y = np.zeros_like(x)
                        xerr = (np.diff(h1d.axes[0]).flatten()/2.)
                        yerr = np.ones_like(y),
                   
                    ax.errorbar(x, y, xerr = xerr, yerr = yerr,
                                    label=f"PNet decay mode = {dm}",
                                    marker='o',
                                    fmt='o',
                                    line=None, color='#2478B7', capsize=4)
                    x_fine = np.linspace(x[0],x[-1],num=30)
                    fitres = fitres_dict[nj][dm]
                    popt = fitres['popt']
                    pcov = fitres['pcov']
                    jac = fitres['jac']
                    def err(x,jac,pcov,popt):
                        from numpy import sqrt,einsum,abs
                        return sqrt(abs(einsum('i,ij,j',jac(x,popt).T,pcov,jac(x,popt))))

                    import functools
                    err_y = list(map(functools.partial(err, jac=jac,pcov=pcov,popt=popt), x_fine))
                    
                    y_fitf = fitres['fitf'](x_fine,*popt)
                    y_fitf_up = fitres['fitf'](x_fine,*popt) + err_y
                    y_fitf_down = fitres['fitf'](x_fine,*(popt)) - err_y
                
                    ax.plot(x_fine,
                            y_fitf,
                            color='#FF867B')
                    ax.fill_between(x_fine, y_fitf_up,  y_fitf_down, color='#83d55f', alpha=0.5)
                    ax.set_ylabel('Fake Factor')
                    ax.set_xlabel('Tau pT [GeV]')
                    ax.set_title(f'Jet Fake Factors : Tau PNet Decay Mode {dm}, Njets {nj}')
                    ax.annotate(rf"$\frac{{\chi^2}}{{ndf}} = \frac{{{np.round(fitres['chi2'],2)}}}{{{fitres['ndf']}}}$",
                                (0.8, 0.75),
                                xycoords='axes fraction',
                                fontsize=20)
                    
                    formula_str = eval_formula(fitres['fitf_str'],popt, make_rounding=True)
                    
                    ax.annotate('y=' + formula_str,
                                (0.01, 0.95),
                                xycoords='axes fraction',
                                fontsize=12)
                    
                    self.output()['plots1d']['_'.join((h_name,str(dm),str(nj)))].dump(fig, formatter="mpl")